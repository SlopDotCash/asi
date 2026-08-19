"""Tests for out-of-hypothesis-class Step 2 benchmark streams.

These streams (``OutOfClassPolynomialStream``, ``FrequencyMismatchStream``,
``CompositionalStream``) generate targets whose minimal representation lies
outside a 1-layer pair-product or tanh feature bank.  The tests here verify
shape correctness, JIT compatibility via ``jax.lax.scan``, and the
out-of-class structural properties that motivate each stream.
"""

import json
import math
import time
from decimal import Decimal
from fractions import Fraction
from numbers import Real

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.out_of_class import (
    CompositionalStream,
    FrequencyMismatchStream,
    OutOfClassPolynomialStream,
)


class _FloatCoercible:
    """Non-real object that would be accepted by ``math.isfinite``."""

    def __float__(self) -> float:
        return 0.5


class _PositiveRatioFloat(float):
    """Negative float subclass whose ratio hook reports a positive value."""

    def as_integer_ratio(self) -> tuple[int, int]:
        return (1, 4)


class _PositiveInt(int):
    """Negative int subclass whose conversion hook reports a positive value."""

    def __int__(self) -> int:
        return 1


class _ExplodingRatioFloat(float):
    """Float subclass whose ratio hook must never execute during validation."""

    def as_integer_ratio(self) -> tuple[int, int]:
        raise RuntimeError("untrusted ratio hook executed")


def _scan_collect(stream, num_steps: int, key) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a stream for ``num_steps`` via ``jax.lax.scan`` and stack outputs.

    Used by the ``test_collect_via_scan`` cases to confirm that each stream
    composes cleanly with ``jax.lax.scan`` (i.e. is JIT-compatible).
    """
    state = stream.init(key)

    def body(carry, idx):
        ts, new_state = stream.step(carry, idx)
        return new_state, (ts.observation, ts.target)

    _, (observations, targets) = jax.lax.scan(
        body, state, jnp.arange(num_steps)
    )
    return observations, targets


# =============================================================================
# OutOfClassPolynomialStream
# =============================================================================


class TestOutOfClassPolynomialStream:
    """Tests for the degree-3 polynomial out-of-class stream."""

    @pytest.mark.parametrize("active_count", [0, -1, True, 1.0, np.int64(1)])
    def test_active_triple_count_requires_positive_builtin_int(
        self, active_count: object
    ) -> None:
        with pytest.raises(ValueError, match="positive built-in integer"):
            OutOfClassPolynomialStream(
                feature_dim=4,
                active_triples_per_context=active_count,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "field",
        ["feature_dim", "n_tasks", "n_contexts", "context_length"],
    )
    @pytest.mark.parametrize("value", [True, False, 1.0, np.int64(3), "3", None])
    def test_dimensions_require_positive_builtin_ints(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("feature_dim", 2),
            ("n_tasks", 0),
            ("n_contexts", 0),
            ("context_length", 0),
        ],
    )
    def test_dimensions_enforce_minimums(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        [
            "feature_dim",
            "n_tasks",
            "n_contexts",
            "context_length",
            "active_triples_per_context",
        ],
    )
    def test_dimensions_reject_values_above_the_int32_domain(self, field: str) -> None:
        with pytest.raises(ValueError, match=rf"{field}.*int32 max"):
            OutOfClassPolynomialStream(**{field: 2**31})  # type: ignore[arg-type]

    def test_context_length_accepts_int32_max_and_runs_eager_and_jit_step(self) -> None:
        maximum = int(np.iinfo(np.int32).max)
        stream = OutOfClassPolynomialStream(
            feature_dim=3,
            n_tasks=1,
            n_contexts=1,
            context_length=maximum,
            active_triples_per_context=1,
            noise_std=0.0,
        )
        state = stream.init(jr.key(98))
        step_index = jnp.array(0, dtype=jnp.int32)

        eager_timestep, _ = stream.step(state, step_index)
        jit_timestep, _ = jax.jit(stream.step)(state, step_index)

        assert stream._context_length == maximum  # noqa: SLF001
        chex.assert_tree_all_finite((eager_timestep, jit_timestep))

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize(
        "value",
        [
            True,
            np.bool_(True),
            "0.5",
            Decimal("0.5"),
            _FloatCoercible(),
            float("nan"),
            float("inf"),
            float("-inf"),
            1e100,
        ],
    )
    def test_scientific_scalars_require_safe_finite_float32_reals(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize(
        "value",
        [
            _PositiveRatioFloat(-0.25),
            _PositiveInt(-1),
            _ExplodingRatioFloat(0.5),
        ],
        ids=("forged-float-ratio", "forged-int-conversion", "raising-float-ratio"),
    )
    def test_scientific_scalars_reject_untrusted_real_subclasses(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    def test_scientific_scalars_do_not_hash_or_compare_untrusted_actual_types(
        self,
        field: str,
    ) -> None:
        calls: list[str] = []

        class HostileNumericMeta(type):
            def __hash__(cls) -> int:
                calls.append("hash")
                raise RuntimeError("untrusted metaclass hash hook executed")

            def __eq__(cls, other: object) -> bool:
                del other
                calls.append("eq")
                raise RuntimeError("untrusted metaclass equality hook executed")

        class HostileFloat(float, metaclass=HostileNumericMeta):
            pass

        calls.clear()
        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(  # type: ignore[arg-type]
                **{field: HostileFloat(1.0)}
            )

        assert calls == []

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize("slot", ["_numerator", "_denominator"])
    def test_scientific_scalars_reject_poisoned_exact_fraction_components_without_hooks(
        self,
        field: str,
        slot: str,
    ) -> None:
        calls = 0

        class ExplodingInt(int):
            def __int__(self) -> int:
                nonlocal calls
                calls += 1
                raise RuntimeError("untrusted Fraction component hook executed")

        value = Fraction(1, 4)
        component = ExplodingInt(1 if slot == "_numerator" else 4)
        object.__setattr__(value, slot, component)

        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

        assert calls == 0

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize("slot", ["_numerator", "_denominator"])
    def test_scientific_scalars_normalize_missing_exact_fraction_slots(
        self,
        field: str,
        slot: str,
    ) -> None:
        value = Fraction(1, 4)
        object.__delattr__(value, slot)

        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize("denominator", [0, -4])
    def test_scientific_scalars_reject_nonpositive_exact_fraction_denominators(
        self,
        field: str,
        denominator: int,
    ) -> None:
        value = Fraction(1, 4)
        object.__setattr__(value, "_denominator", denominator)

        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "scalar_type",
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
            np.float16,
            np.float32,
            np.float64,
            np.longdouble,
        ],
    )
    def test_scientific_scalars_accept_supported_numpy_types(
        self,
        scalar_type: type[np.generic],
    ) -> None:
        value = scalar_type(1)
        stream = OutOfClassPolynomialStream(
            feature_std=value,
            linear_scale=value,
            noise_std=value,
        )

        assert stream._feature_std == 1.0  # noqa: SLF001
        assert stream._linear_scale == 1.0  # noqa: SLF001
        assert stream._noise_std == 1.0  # noqa: SLF001

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize("scalar_type", [np.longlong, np.ulonglong])
    def test_scientific_scalars_preserve_distinct_numpy_long_integer_types(
        self,
        field: str,
        scalar_type: type[np.generic],
    ) -> None:
        stream = OutOfClassPolynomialStream(  # type: ignore[arg-type]
            **{field: scalar_type(1)}
        )

        assert getattr(stream, f"_{field}") == 1.0

    @pytest.mark.parametrize("field", ["feature_std", "noise_std"])
    @pytest.mark.parametrize("value", [-1.0, -Fraction(1, 2**200)])
    def test_standard_deviations_reject_negative_exact_reals(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

    def test_noise_std_rejects_non_real_with_a_real_class_facade(self) -> None:
        class RealFacade:
            @property
            def __class__(self) -> type[float]:
                return float

            def as_integer_ratio(self) -> tuple[int, int]:
                return (1, 2)

        value = RealFacade()
        assert isinstance(value, Real)
        assert not issubclass(type(value), Real)

        with pytest.raises(ValueError, match="noise_std"):
            OutOfClassPolynomialStream(noise_std=value)  # type: ignore[arg-type]

    def test_noise_std_rejects_ratio_component_with_an_integral_class_facade(
        self,
    ) -> None:
        class IntegralFacade:
            @property
            def __class__(self) -> type[int]:
                return int

            def __int__(self) -> int:
                return 1

        class MalformedRatioFloat(float):
            def as_integer_ratio(self) -> tuple[object, int]:
                return (IntegralFacade(), 2)

        with pytest.raises(ValueError, match="noise_std"):
            OutOfClassPolynomialStream(noise_std=MalformedRatioFloat(0.5))

    def test_noise_std_rejects_negative_exact_ratio_from_float_subclass(self) -> None:
        class NegativeRatioFloat(float):
            def as_integer_ratio(self) -> tuple[int, int]:
                return (-1, 2**200)

        value = NegativeRatioFloat(0.5)
        assert value >= 0.0
        assert value.as_integer_ratio()[0] < 0

        with pytest.raises(ValueError, match="noise_std"):
            OutOfClassPolynomialStream(noise_std=value)

    @pytest.mark.parametrize("field", ["feature_std", "linear_scale", "noise_std"])
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            (Fraction(-1, 2**60), float(np.float32(1.0))),
            (Fraction(0), float(np.float32(1.0))),
            (
                Fraction(1, 2**60),
                float(np.nextafter(np.float32(1.0), np.float32(2.0))),
            ),
        ],
        ids=("below", "tie-to-even", "above"),
    )
    def test_scientific_scalars_round_exact_midpoints_once(
        self,
        field: str,
        offset: Fraction,
        expected: float,
    ) -> None:
        value = Fraction(1) + Fraction(1, 2**24) + offset
        stream = OutOfClassPolynomialStream(**{field: value})  # type: ignore[arg-type]

        assert getattr(stream, f"_{field}") == expected

    def test_multiplier_products_retain_float32_headroom(self) -> None:
        multiplier_max = float(np.sqrt(np.finfo(np.float32).max))

        with pytest.raises(ValueError, match="feature_std cubed"):
            OutOfClassPolynomialStream(feature_std=3_000_000.0)
        with pytest.raises(ValueError, match="linear_scale and feature_std"):
            OutOfClassPolynomialStream(
                feature_std=2.0,
                linear_scale=multiplier_max,
            )

    @pytest.mark.parametrize("value", [0, 1, "yes", np.bool_(True)])
    def test_include_squares_requires_strict_bool(self, value: object) -> None:
        with pytest.raises(ValueError, match="include_squares"):
            OutOfClassPolynomialStream(include_squares=value)  # type: ignore[arg-type]

    def test_include_squares_rejects_invalid_type_without_calling_repr(self) -> None:
        repr_calls = 0

        class ExplodingRepr:
            def __repr__(self) -> str:
                nonlocal repr_calls
                repr_calls += 1
                raise RuntimeError("untrusted repr hook executed")

        with pytest.raises(ValueError, match="include_squares"):
            OutOfClassPolynomialStream(include_squares=ExplodingRepr())  # type: ignore[arg-type]

        assert repr_calls == 0

    def test_legal_scalar_endpoints_remain_supported(self) -> None:
        stream = OutOfClassPolynomialStream(
            feature_dim=3,
            n_tasks=1,
            n_contexts=1,
            context_length=1,
            feature_std=0.0,
            linear_scale=-0.0,
            noise_std=0.0,
            include_squares=False,
        )

        assert stream.feature_dim == 3
        assert math.copysign(1.0, stream._linear_scale) == -1.0  # noqa: SLF001
        timestep, _ = stream.step(stream.init(jr.key(97)), jnp.array(0))
        chex.assert_tree_all_finite((timestep.observation, timestep.target))

    def test_active_triple_count_caps_at_available_triples(self) -> None:
        stream = OutOfClassPolynomialStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            active_triples_per_context=100,
        )
        state = stream.init(jr.key(92))

        assert state.context_weights.shape == (2, 2, 4)
        assert int(jnp.count_nonzero(state.context_weights)) == state.context_weights.size

    def test_unique_finite_scores_preserve_legacy_context_weights(self) -> None:
        stream = OutOfClassPolynomialStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            active_triples_per_context=2,
        )
        root_key = jr.key(315)
        state = stream.init(root_key)
        _, context_key, mask_key, _ = jr.split(root_key, 4)
        triple_count = state.triples_left.shape[0]
        dense_weights = jr.normal(
            context_key,
            (2, 2, triple_count),
            dtype=jnp.float32,
        )
        scores = jr.uniform(mask_key, (2, 2, triple_count), dtype=jnp.float32)
        for row in np.asarray(scores).reshape((-1, triple_count)):
            assert np.unique(row).size == triple_count
        threshold = jnp.sort(scores, axis=-1)[..., 1:2]
        legacy_mask = scores <= threshold
        expected = dense_weights * legacy_mask.astype(jnp.float32) / jnp.sqrt(
            jnp.sum(legacy_mask, axis=-1, keepdims=True)
        )

        np.testing.assert_array_equal(
            np.asarray(state.context_weights),
            np.asarray(expected),
        )

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize(
        ("scores", "expected_mask"),
        [
            ([0.5, 0.5, 0.5, 0.5], [True, True, False, False]),
            ([0.5, 0.5, 0.2, 0.9], [True, False, True, False]),
        ],
    )
    def test_init_selects_exact_stable_active_count_under_ties(
        self,
        monkeypatch: pytest.MonkeyPatch,
        compiled: bool,
        scores: list[float],
        expected_mask: list[bool],
    ) -> None:
        stream = OutOfClassPolynomialStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            active_triples_per_context=2,
            noise_std=0.0,
        )

        def fixed_normal(key, shape, dtype=jnp.float32, **kwargs):
            del key, kwargs
            return jnp.ones(shape, dtype=dtype)

        def fixed_uniform(key, shape, dtype=jnp.float32, **kwargs):
            del key, kwargs
            return jnp.broadcast_to(jnp.asarray(scores, dtype=dtype), shape)

        monkeypatch.setattr(
            "alberta_framework.streams.out_of_class.jr.normal", fixed_normal
        )
        monkeypatch.setattr(
            "alberta_framework.streams.out_of_class.jr.uniform", fixed_uniform
        )
        init = jax.jit(stream.init) if compiled else stream.init
        state = init(jr.key(0))

        actual_mask = state.context_weights != 0.0
        expected = jnp.broadcast_to(
            jnp.asarray(expected_mask), state.context_weights.shape
        )
        chex.assert_trees_all_equal(actual_mask, expected)

        observations, targets = jax.jit(
            lambda key: _scan_collect(stream, num_steps=4, key=key)
        )(jr.key(1))
        chex.assert_tree_all_finite((observations, targets))

    def test_step_shapes(self):
        stream = OutOfClassPolynomialStream(
            feature_dim=5,
            n_tasks=3,
            n_contexts=2,
            context_length=4,
            active_triples_per_context=2,
        )
        state = stream.init(jr.key(0))
        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (5,))
        chex.assert_shape(timestep.target, (3,))
        assert int(new_state.step_count) == 1

    def test_finite_outputs(self):
        stream = OutOfClassPolynomialStream(
            feature_dim=6,
            n_tasks=2,
            active_triples_per_context=3,
        )
        state = stream.init(jr.key(1))
        timestep, _ = stream.step(state, jnp.array(0))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_collect_via_scan(self):
        stream = OutOfClassPolynomialStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            context_length=3,
            active_triples_per_context=2,
        )
        observations, targets = _scan_collect(stream, num_steps=12, key=jr.key(2))
        chex.assert_shape(observations, (12, 4))
        chex.assert_shape(targets, (12, 2))
        chex.assert_tree_all_finite(observations)
        chex.assert_tree_all_finite(targets)

    def test_higher_order_structure_present(self):
        """Variance of targets should grow faster than O(scale^2).

        For a pure linear oracle, ``var(y(scale*x)) = scale^2 var(y(x))``.
        For our degree-3 oracle (with a small linear component), variance
        should grow much faster than that across a range of input scales.

        We feed inputs of growing scale, measure target variance per task,
        and verify that the empirical variance ratio (largest_scale vs
        baseline_scale) substantially exceeds the ratio expected of a
        purely linear generator.
        """
        # Build a stream with the noise turned down so the polynomial
        # signal dominates the variance estimate, and with a tiny linear
        # component so the higher-order effect is visible.
        stream = OutOfClassPolynomialStream(
            feature_dim=5,
            n_tasks=2,
            n_contexts=1,
            context_length=10_000,
            active_triples_per_context=4,
            linear_scale=0.0,
            noise_std=0.0,
        )
        state = stream.init(jr.key(3))
        n_samples = 256
        scales = jnp.array([0.5, 1.0, 2.0, 3.0], dtype=jnp.float32)

        # Manually evaluate the deterministic polynomial part (no noise,
        # no linear) on Gaussian samples scaled by each ``scale`` value.
        base_x = jr.normal(jr.key(4), (n_samples, 5), dtype=jnp.float32)

        def target_for_scale(s: jnp.ndarray) -> jnp.ndarray:
            xs = s * base_x  # (n_samples, feature_dim)
            triples = (
                xs[:, state.triples_left]
                * xs[:, state.triples_middle]
                * xs[:, state.triples_right]
            )  # (n_samples, n_triples)
            ws = state.context_weights[0]  # (n_tasks, n_triples)
            return triples @ ws.T  # (n_samples, n_tasks)

        # Per-task variance at each scale, summed across tasks.
        variances = jnp.array(
            [jnp.var(target_for_scale(s)).item() for s in scales]
        )
        # For a degree-3 polynomial, variance scales like s^6 (since each
        # output is a sum of triple products and var grows as the square
        # of the output magnitude scaling).  For a linear oracle it grows
        # as s^2.  We require that the ratio var(scale=3.0) / var(scale=1.0)
        # be far above 9 (the linear bound).
        ratio = variances[3] / variances[1]
        assert ratio > 9.0 * 50.0, (
            f"Expected target variance to scale super-linearly with input"
            f" magnitude (ratio at scale=3 vs scale=1 was {ratio:.2f},"
            f" linear bound is 9.0). All variances: {variances.tolist()}"
        )


# =============================================================================
# FrequencyMismatchStream
# =============================================================================


class TestFrequencyMismatchStream:
    """Tests for the trigonometric out-of-class stream."""

    @pytest.mark.parametrize(
        "field",
        [
            "feature_dim",
            "n_tasks",
            "n_components_per_task",
            "n_contexts",
            "context_length",
        ],
    )
    @pytest.mark.parametrize("value", [True, False, 1.0, np.int64(3), "3", None])
    def test_dimensions_require_positive_builtin_ints(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            FrequencyMismatchStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        [
            "feature_dim",
            "n_tasks",
            "n_components_per_task",
            "n_contexts",
            "context_length",
        ],
    )
    def test_dimensions_reject_values_above_the_int32_domain(self, field: str) -> None:
        with pytest.raises(ValueError, match=rf"{field}.*int32 max"):
            FrequencyMismatchStream(**{field: 2**31})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["amplitude_scale", "noise_std"])
    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), -1.0, -Fraction(1, 2**200)],
    )
    def test_scale_parameters_reject_nonfinite_or_negative_exact_reals(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            FrequencyMismatchStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["amplitude_scale", "noise_std"])
    def test_scale_parameters_reject_untrusted_real_subclasses(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            FrequencyMismatchStream(  # type: ignore[arg-type]
                **{field: _PositiveRatioFloat(-0.5)}
            )

    def test_scale_parameters_are_canonical_float32_values(self) -> None:
        stream = FrequencyMismatchStream(
            amplitude_scale=Fraction(1, 3),
            noise_std=np.float64(0.2),
        )
        assert type(stream._amplitude_scale) is float  # noqa: SLF001
        assert type(stream._noise_std) is float  # noqa: SLF001
        assert stream._amplitude_scale == float(np.float32(1 / 3))  # noqa: SLF001
        assert stream._noise_std == float(np.float32(0.2))  # noqa: SLF001

    @pytest.mark.parametrize(
        ("omega_min", "omega_max"),
        [(float("nan"), 3.0), (0.5, float("inf")), (float("inf"), float("inf"))],
    )
    def test_frequency_bounds_must_be_finite(
        self, omega_min: float, omega_max: float
    ) -> None:
        with pytest.raises(ValueError):
            FrequencyMismatchStream(omega_min=omega_min, omega_max=omega_max)

    @pytest.mark.parametrize(
        ("omega_min", "omega_max"),
        [(1e-50, 1.0), (1.0, 1e100), (1.0, 1.0 + 1e-12)],
    )
    def test_frequency_bounds_must_define_a_positive_float32_interval(
        self, omega_min: float, omega_max: float
    ) -> None:
        with pytest.raises(ValueError, match="float32"):
            FrequencyMismatchStream(omega_min=omega_min, omega_max=omega_max)

    @pytest.mark.parametrize("value", [True, "0.5", object()])
    def test_frequency_bounds_reject_non_real_values(self, value: object) -> None:
        with pytest.raises(ValueError, match="float32"):
            FrequencyMismatchStream(omega_min=value)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="float32"):
            FrequencyMismatchStream(omega_max=value)  # type: ignore[arg-type]

    def test_frequency_bounds_are_canonicalized_to_float32(self) -> None:
        stream = FrequencyMismatchStream(omega_min=0.7, omega_max=2.3)
        state = stream.init(jr.key(23))

        expected_min = jnp.asarray(0.7, dtype=jnp.float32)
        expected_max = jnp.asarray(2.3, dtype=jnp.float32)
        assert bool(jnp.all(jnp.isfinite(state.omegas)))
        assert bool(jnp.all(state.omegas >= expected_min))
        assert bool(jnp.all(state.omegas < expected_max))

    def test_frequency_bounds_narrow_the_original_real_once(self) -> None:
        midpoint_plus = Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60)
        directly_rounded = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        assert directly_rounded != float(np.float32(float(midpoint_plus)))

        stream = FrequencyMismatchStream(omega_min=1.0, omega_max=midpoint_plus)
        assert stream._omega_max == directly_rounded  # noqa: SLF001

    @pytest.mark.parametrize(
        ("omega_max", "expected"),
        [
            (
                Fraction(1, 1) + Fraction(1, 2**24) - Fraction(1, 2**60),
                None,
            ),
            (Fraction(1, 1) + Fraction(1, 2**24), None),
            (
                Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60),
                float(np.nextafter(np.float32(1.0), np.float32(2.0))),
            ),
        ],
        ids=("below", "tie-to-even", "above"),
    )
    def test_frequency_bounds_round_fraction_midpoints_once(
        self,
        omega_max: Fraction,
        expected: float | None,
    ) -> None:
        if expected is None:
            with pytest.raises(ValueError, match="omega_max must exceed"):
                FrequencyMismatchStream(omega_min=1.0, omega_max=omega_max)
            return

        stream = FrequencyMismatchStream(omega_min=1.0, omega_max=omega_max)
        assert stream._omega_max == expected  # noqa: SLF001

    def test_frequency_bounds_apply_exact_float32_overflow_midpoint(self) -> None:
        float32_max = (2**24 - 1) * 2**104
        overflow_midpoint = float32_max + 2**103

        stream = FrequencyMismatchStream(
            omega_min=1.0,
            omega_max=Fraction(overflow_midpoint - 1),
        )
        assert stream._omega_max == float(np.finfo(np.float32).max)  # noqa: SLF001
        with pytest.raises(ValueError, match="omega_max"):
            FrequencyMismatchStream(
                omega_min=1.0,
                omega_max=Fraction(overflow_midpoint),
            )

    def test_frequency_bounds_apply_exact_subnormal_midpoint(self) -> None:
        subnormal_midpoint = Fraction(1, 2**150)

        with pytest.raises(ValueError, match="omega_min"):
            FrequencyMismatchStream(omega_min=subnormal_midpoint, omega_max=1.0)
        stream = FrequencyMismatchStream(
            omega_min=subnormal_midpoint + Fraction(1, 2**200),
            omega_max=1.0,
        )
        assert stream._omega_min == float(  # noqa: SLF001
            np.nextafter(np.float32(0.0), np.float32(1.0))
        )

    def test_frequency_bounds_reject_negative_zero(self) -> None:
        with pytest.raises(ValueError, match="omega_min"):
            FrequencyMismatchStream(omega_min=-0.0, omega_max=1.0)

    def test_step_shapes(self):
        stream = FrequencyMismatchStream(
            feature_dim=4,
            n_tasks=2,
            n_components_per_task=3,
            n_contexts=2,
            context_length=4,
        )
        state = stream.init(jr.key(0))
        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (4,))
        chex.assert_shape(timestep.target, (2,))
        assert int(new_state.step_count) == 1

    def test_finite_outputs(self):
        stream = FrequencyMismatchStream(
            feature_dim=3,
            n_tasks=2,
            n_components_per_task=4,
        )
        state = stream.init(jr.key(1))
        timestep, _ = stream.step(state, jnp.array(0))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_collect_via_scan(self):
        stream = FrequencyMismatchStream(
            feature_dim=3,
            n_tasks=2,
            n_components_per_task=2,
            n_contexts=2,
            context_length=3,
        )
        observations, targets = _scan_collect(stream, num_steps=10, key=jr.key(2))
        chex.assert_shape(observations, (10, 3))
        chex.assert_shape(targets, (10, 2))
        chex.assert_tree_all_finite(observations)
        chex.assert_tree_all_finite(targets)

    def test_periodic_structure(self):
        """Targets should oscillate when sweeping along one input dim.

        For a sinusoidal oracle, sweeping a single input dimension across
        ``[-pi, pi]`` produces a target that crosses zero multiple times
        (i.e. is non-monotonic).  We construct an oracle that forces the
        first task's first component to listen on dim 0 and have large
        amplitude, then sweep dim 0 along ``[-pi, pi]`` and count sign
        changes of the target.
        """
        stream = FrequencyMismatchStream(
            feature_dim=2,
            n_tasks=1,
            n_components_per_task=1,
            n_contexts=1,
            context_length=10_000,
            omega_min=2.0,
            omega_max=2.001,
            amplitude_scale=2.0,
            noise_std=0.0,
        )
        state = stream.init(jr.key(7))
        # Override the active_indices and amplitudes deterministically so
        # we don't depend on RNG to land on dim 0 / nonzero amplitude.
        active_indices = state.active_indices.at[:].set(0)
        amplitudes = state.amplitudes.at[:].set(2.0)
        state = state.replace(  # type: ignore[attr-defined]
            active_indices=active_indices,
            amplitudes=amplitudes,
        )

        sweep = jnp.linspace(-jnp.pi, jnp.pi, 100, dtype=jnp.float32)
        omegas = state.omegas[0, 0, 0]
        phases = state.phases[0, 0, 0]
        targets = amplitudes[0, 0, 0] * jnp.sin(omegas * sweep + phases)

        # Count sign changes.  For omega ~ 2.0 over [-pi, pi] (range 2*pi),
        # the sinusoid completes ~2 full cycles and crosses zero ~4 times,
        # so we require at least 3 sign changes.
        signs = jnp.sign(targets)
        sign_changes = int(jnp.sum(jnp.abs(jnp.diff(signs)) > 0))
        assert sign_changes >= 3, (
            f"Expected sweeping a single input dim to produce multiple sign"
            f" changes in a sinusoidal oracle (got {sign_changes})"
        )


# =============================================================================
# CompositionalStream
# =============================================================================


class TestCompositionalStream:
    """Tests for the 2-hidden-layer compositional out-of-class stream."""

    @pytest.mark.parametrize(
        "field",
        [
            "feature_dim",
            "n_tasks",
            "inner_hidden",
            "outer_components",
            "n_contexts",
            "context_length",
        ],
    )
    @pytest.mark.parametrize("value", [True, False, 1.0, np.int64(3), "3", None])
    def test_dimensions_require_positive_builtin_ints(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            CompositionalStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        [
            "feature_dim",
            "n_tasks",
            "inner_hidden",
            "outer_components",
            "n_contexts",
            "context_length",
        ],
    )
    def test_dimensions_reject_values_above_the_int32_domain(self, field: str) -> None:
        with pytest.raises(ValueError, match=rf"{field}.*int32 max"):
            CompositionalStream(**{field: 2**31})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["feature_std", "amplitude_scale", "noise_std"])
    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), -1.0, -Fraction(1, 2**200)],
    )
    def test_scale_parameters_reject_nonfinite_or_negative_exact_reals(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            CompositionalStream(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["feature_std", "amplitude_scale", "noise_std"])
    def test_scale_parameters_reject_untrusted_real_subclasses(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            CompositionalStream(**{field: _PositiveRatioFloat(-0.5)})  # type: ignore[arg-type]

    def test_scale_parameters_are_canonical_float32_values(self) -> None:
        stream = CompositionalStream(
            feature_std=Fraction(1, 3),
            amplitude_scale=np.float64(0.2),
            noise_std=np.float32(0.1),
        )
        assert type(stream._feature_std) is float  # noqa: SLF001
        assert type(stream._amplitude_scale) is float  # noqa: SLF001
        assert type(stream._noise_std) is float  # noqa: SLF001

    def test_feature_and_weight_scale_reject_unsafe_float32_product(self) -> None:
        safe = float(np.sqrt(np.finfo(np.float32).max, dtype=np.float32))
        with pytest.raises(ValueError, match="feature_std and weight_scale"):
            CompositionalStream(feature_std=safe, weight_scale=safe)

    @pytest.mark.parametrize("weight_scale", [float("nan"), float("inf"), float("-inf")])
    def test_weight_scale_must_be_finite(self, weight_scale: float) -> None:
        with pytest.raises(ValueError, match="weight_scale must be finite"):
            CompositionalStream(weight_scale=weight_scale)

    @pytest.mark.parametrize(
        "weight_scale",
        [
            True,
            False,
            np.bool_(True),
            "0.5",
            Decimal("0.5"),
            _FloatCoercible(),
            object(),
        ],
    )
    def test_weight_scale_rejects_bool_non_real_and_coercive_inputs(
        self, weight_scale: object
    ) -> None:
        with pytest.raises(ValueError, match="weight_scale must be finite in float32"):
            CompositionalStream(weight_scale=weight_scale)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "weight_scale",
        [Fraction(1, 2), np.float32(0.5), np.float64(0.5), np.int64(2)],
    )
    def test_weight_scale_is_stored_as_a_json_safe_canonical_float32_float(
        self, weight_scale: object
    ) -> None:
        stream = CompositionalStream(weight_scale=weight_scale)  # type: ignore[arg-type]
        expected = float(np.float32(float(weight_scale)))

        assert type(stream._weight_scale) is float
        assert stream._weight_scale == expected
        encoded = json.dumps({"weight_scale": stream._weight_scale}, allow_nan=False)
        assert json.loads(encoded) == {"weight_scale": expected}
        state = stream.init(jr.key(93))
        assert state.inner_w.dtype == jnp.float32
        assert state.outer_w.dtype == jnp.float32

    def test_weight_scale_narrows_the_original_real_once(self) -> None:
        midpoint_plus = Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60)
        directly_rounded = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        assert directly_rounded != float(np.float32(float(midpoint_plus)))

        stream = CompositionalStream(weight_scale=midpoint_plus)
        assert stream._weight_scale == directly_rounded

    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            (Fraction(-1, 1 << 60), 1.0),
            (Fraction(0), 1.0),
            (
                Fraction(1, 1 << 60),
                float(np.nextafter(np.float32(1.0), np.float32(2.0))),
            ),
        ],
        ids=["below", "tie", "above"],
    )
    def test_weight_scale_rounds_exact_fraction_midpoints_once(
        self, offset: Fraction, expected: float
    ) -> None:
        midpoint = Fraction(1) + Fraction(1, 1 << 24)
        stream = CompositionalStream(weight_scale=midpoint + offset)

        assert stream._weight_scale == expected

    def test_weight_scale_fraction_midpoint_uses_ties_to_even(self) -> None:
        lower = np.nextafter(np.float32(1.0), np.float32(2.0))
        upper = np.nextafter(lower, np.float32(2.0))
        lower_ratio = Fraction(*lower.as_integer_ratio())
        upper_ratio = Fraction(*upper.as_integer_ratio())
        midpoint = (lower_ratio + upper_ratio) / 2

        stream = CompositionalStream(weight_scale=midpoint)
        assert stream._weight_scale == float(upper)

    @pytest.mark.parametrize("weight_scale", [1e39, -1e39, 2e38, -2e38])
    def test_weight_scale_rejects_float32_conversion_or_initialization_overflow(
        self, weight_scale: float
    ) -> None:
        with pytest.raises(ValueError, match="weight_scale must be finite in float32"):
            CompositionalStream(weight_scale=weight_scale)

    def test_weight_scale_safe_float32_boundary(self) -> None:
        safe = np.sqrt(np.finfo(np.float32).max, dtype=np.float32)
        unsafe = np.nextafter(safe, np.float32(np.inf), dtype=np.float32)

        stream = CompositionalStream(weight_scale=float(safe))
        assert stream._weight_scale == float(safe)
        with pytest.raises(ValueError, match="weight_scale must be finite in float32"):
            CompositionalStream(weight_scale=float(unsafe))

    @pytest.mark.parametrize("weight_scale", [0.0, -0.0, -2.5])
    def test_weight_scale_preserves_valid_zero_and_negative_semantics(
        self, weight_scale: float
    ) -> None:
        stream = CompositionalStream(weight_scale=weight_scale)

        assert stream._weight_scale == weight_scale
        assert math.copysign(1.0, stream._weight_scale) == math.copysign(
            1.0, weight_scale
        )
        eager_state = stream.init(jr.key(94))
        eager_timestep, _ = stream.step(eager_state, jnp.array(0, dtype=jnp.int32))
        jit_state = jax.jit(stream.init)(jr.key(94))
        jit_timestep, _ = jax.jit(stream.step)(jit_state, jnp.array(0, dtype=jnp.int32))
        chex.assert_tree_all_finite(
            (
                eager_state.inner_w,
                eager_state.outer_w,
                eager_timestep.observation,
                eager_timestep.target,
                jit_state.inner_w,
                jit_state.outer_w,
                jit_timestep.observation,
                jit_timestep.target,
            )
        )

    @pytest.mark.parametrize("weight_scale", [1e-50, -1e-50])
    def test_weight_scale_float32_underflow_is_canonical_signed_zero(
        self, weight_scale: float
    ) -> None:
        stream = CompositionalStream(weight_scale=weight_scale)
        zero_stream = CompositionalStream(weight_scale=math.copysign(0.0, weight_scale))

        assert stream._weight_scale == 0.0
        assert math.copysign(1.0, stream._weight_scale) == math.copysign(
            1.0, weight_scale
        )
        chex.assert_trees_all_equal(
            stream.init(jr.key(95)), zero_stream.init(jr.key(95))
        )

    @pytest.mark.parametrize("sign", [1, -1])
    @pytest.mark.parametrize(
        ("offset", "expected_magnitude"),
        [
            (Fraction(-1, 1 << 200), 0.0),
            (Fraction(0), 0.0),
            (Fraction(1, 1 << 200), float(np.nextafter(np.float32(0.0), 1.0))),
        ],
        ids=["below", "tie", "above"],
    )
    def test_weight_scale_rounds_half_minimum_subnormal_to_even_signed_zero(
        self, sign: int, offset: Fraction, expected_magnitude: float
    ) -> None:
        half_minimum_subnormal = Fraction(1, 1 << 150)
        weight_scale = sign * (half_minimum_subnormal + offset)

        stream = CompositionalStream(weight_scale=weight_scale)

        assert abs(stream._weight_scale) == expected_magnitude
        assert math.copysign(1.0, stream._weight_scale) == float(sign)

    def test_weight_scale_boundary_stays_finite_eager_and_jit_across_seeds(
        self,
    ) -> None:
        safe = float(np.sqrt(np.finfo(np.float32).max, dtype=np.float32))
        stream = CompositionalStream(weight_scale=safe)
        jit_init = jax.jit(stream.init)
        jit_step = jax.jit(stream.step)

        for seed in range(16):
            key = jr.key(seed)
            eager_state = stream.init(key)
            eager_timestep, _ = stream.step(
                eager_state, jnp.array(0, dtype=jnp.int32)
            )
            jit_state = jit_init(key)
            jit_timestep, _ = jit_step(jit_state, jnp.array(0, dtype=jnp.int32))
            chex.assert_tree_all_finite(
                (
                    eager_state.inner_w,
                    eager_state.outer_w,
                    eager_timestep.target,
                    jit_state.inner_w,
                    jit_state.outer_w,
                    jit_timestep.target,
                )
            )
            np.testing.assert_allclose(
                np.asarray(eager_timestep.target),
                np.asarray(jit_timestep.target),
                rtol=1e-5,
                atol=1e-6,
            )

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_numpy_weight_scale_is_x64_invariant(self, use_jit: bool) -> None:
        def run(x64_enabled: bool):
            with jax.enable_x64(x64_enabled):
                stream = CompositionalStream(weight_scale=np.float64(0.7))
                init = jax.jit(stream.init) if use_jit else stream.init
                step = jax.jit(stream.step) if use_jit else stream.step
                state = init(jr.key(96))
                timestep, _ = step(state, jnp.array(0, dtype=jnp.int32))
                return stream, state, timestep

        stream32, state32, timestep32 = run(False)
        stream64, state64, timestep64 = run(True)

        assert type(stream32._weight_scale) is float
        assert type(stream64._weight_scale) is float
        assert state32.inner_w.dtype == jnp.float32
        assert state64.inner_w.dtype == jnp.float32
        chex.assert_trees_all_equal(state32, state64)
        chex.assert_trees_all_equal(timestep32, timestep64)

    def test_step_shapes(self):
        stream = CompositionalStream(
            feature_dim=6,
            n_tasks=3,
            inner_hidden=4,
            outer_components=5,
            n_contexts=2,
            context_length=4,
        )
        state = stream.init(jr.key(0))
        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (6,))
        chex.assert_shape(timestep.target, (3,))
        assert int(new_state.step_count) == 1

    def test_finite_outputs(self):
        stream = CompositionalStream(
            feature_dim=4,
            n_tasks=2,
            inner_hidden=3,
            outer_components=3,
        )
        state = stream.init(jr.key(1))
        timestep, _ = stream.step(state, jnp.array(0))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_collect_via_scan(self):
        stream = CompositionalStream(
            feature_dim=5,
            n_tasks=2,
            inner_hidden=3,
            outer_components=3,
            n_contexts=2,
            context_length=3,
        )
        observations, targets = _scan_collect(stream, num_steps=10, key=jr.key(2))
        chex.assert_shape(observations, (10, 5))
        chex.assert_shape(targets, (10, 2))
        chex.assert_tree_all_finite(observations)
        chex.assert_tree_all_finite(targets)

    def test_two_layer_nonlinearity(self):
        """A linear regression should leave substantial residual.

        We collect a batch of samples from the compositional oracle, fit
        a least-squares linear model ``y ~ Wx + b``, and require that the
        residual variance be a non-trivial fraction of the target
        variance.  A purely linear oracle would yield residual / target
        variance close to 0; a linear-plus-noise oracle would only leave
        the noise floor.  A 2-hidden-layer tanh oracle leaves much more.

        We use an aggressive weight scale and an input scale large enough
        that the inner-layer pre-activations push the tanh into its
        nonlinear regime; otherwise tanh acts approximately like the
        identity on small inputs and the oracle collapses toward linear.
        """
        stream = CompositionalStream(
            feature_dim=5,
            n_tasks=1,
            inner_hidden=4,
            outer_components=8,
            n_contexts=1,
            context_length=10_000,
            feature_std=2.0,
            weight_scale=5.0,
            amplitude_scale=2.0,
            noise_std=0.0,
        )
        observations, targets = _scan_collect(
            stream, num_steps=800, key=jr.key(2)
        )

        # Solve y ~ Wx + b via stacking a bias column and lstsq.
        n = observations.shape[0]
        x_bias = jnp.concatenate(
            [observations, jnp.ones((n, 1), dtype=observations.dtype)], axis=1
        )
        # Use jnp.linalg.lstsq for a linear fit.  ``targets`` is (n, 1).
        sol, _, _, _ = jnp.linalg.lstsq(x_bias, targets, rcond=None)
        predictions = x_bias @ sol
        residual = targets - predictions

        target_var = float(jnp.var(targets))
        residual_var = float(jnp.var(residual))
        assert target_var > 0.0, "Target variance should be positive"
        ratio = residual_var / target_var
        # A linear-only oracle would give ratio ~ 0.0 (subject to lstsq
        # numerical floor); 20% residual is well above any plausible
        # numerical residual and strongly indicates the oracle is out of
        # the linear hypothesis class.
        assert ratio > 0.20, (
            f"Linear regression residual variance ratio {ratio:.3f} is too"
            f" small for a compositional oracle; target is out of class"
            f" only if a linear fit leaves substantial residual."
        )


@pytest.mark.parametrize(
    "stream",
    [
        OutOfClassPolynomialStream(
            feature_dim=6,
            n_tasks=3,
            n_contexts=2,
            active_triples_per_context=4,
        ),
        OutOfClassPolynomialStream(
            feature_dim=6,
            n_tasks=3,
            n_contexts=2,
            active_triples_per_context=4,
            include_squares=True,
        ),
        FrequencyMismatchStream(
            feature_dim=3,
            n_tasks=2,
            n_components_per_task=2,
            n_contexts=2,
        ),
        CompositionalStream(
            feature_dim=3,
            n_tasks=2,
            inner_hidden=2,
            outer_components=2,
            n_contexts=2,
        ),
    ],
)
def test_out_of_class_resource_budget_matches_resident_state(stream: object) -> None:
    state = stream.init(jr.key(0))  # type: ignore[attr-defined]
    actual_bytes = sum(int(leaf.nbytes) for leaf in jax.tree.leaves(state))
    budget = stream.resource_budget  # type: ignore[attr-defined]
    assert budget["state_bytes"] == actual_bytes
    assert budget["state_scalars"] * 4 == actual_bytes


def test_out_of_class_derived_state_budgets_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="frequency-mismatch.*64 MiB"):
        FrequencyMismatchStream(
            feature_dim=1,
            n_tasks=1,
            n_components_per_task=1,
            n_contexts=4_194_304,
        )
    with pytest.raises(ValueError, match="compositional.*64 MiB"):
        CompositionalStream(
            feature_dim=1_000,
            n_tasks=1,
            inner_hidden=1_000,
            outer_components=100,
            n_contexts=1,
        )
    with pytest.raises(ValueError, match="out-of-class-polynomial.*64 MiB"):
        OutOfClassPolynomialStream(
            feature_dim=274,
            n_tasks=1,
            n_contexts=2,
            active_triples_per_context=1,
        )


def test_out_of_class_polynomial_rejects_hang_inducing_feature_dim_before_enumeration() -> None:
    """A ``feature_dim`` well inside the int32 domain must not reach ``_triples()``.

    ``_triples()`` enumerates all oracle triples in plain Python --
    O(feature_dim ** 3) work -- before any JAX array is built. Before this
    fix, ``feature_dim`` was only bounded by int32 max, so a caller-supplied
    value of a few thousand (let alone 100_000) would hang the process for an
    effectively unbounded amount of time; independently timed, the pure
    enumeration alone already takes several seconds at ``feature_dim=500``
    and grows cubically from there. The analytic ``math.comb`` precheck in
    ``_polynomial_state_budget`` must reject this immediately, without ever
    calling ``_triples()``.
    """
    start = time.monotonic()
    with pytest.raises(ValueError, match="out-of-class-polynomial"):
        OutOfClassPolynomialStream(feature_dim=100_000, n_tasks=3, n_contexts=4)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, (
        f"rejection took {elapsed:.3f}s; expected an O(1) analytic precheck, "
        "not an attempt to enumerate feature_dim**3 triples"
    )


@pytest.mark.parametrize(
    "stream",
    [FrequencyMismatchStream(), CompositionalStream()],
)
def test_out_of_class_step_clocks_saturate(stream: object) -> None:
    state = stream.init(jr.key(1)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32)
    )
    _, advanced = stream.step(state, jnp.asarray(0, dtype=jnp.int32))  # type: ignore[attr-defined]
    assert int(advanced.step_count) == 2**31 - 1
