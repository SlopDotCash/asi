"""Tests for the MultiHeadMLPLearner and multi-head learning loops."""

import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import (
    LMS,
    AGCBounding,
    Autostep,
    AutostepGTDLambda,
    BatchedMultiHeadResult,
    EMANormalizer,
    MultiHeadLearningResult,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    MultiHeadMLPUpdateResult,
    ObGD,
    ObGDBounding,
    TraceMode,
    WelfordNormalizer,
    multi_head_metrics_to_dicts,
    run_multi_head_learning_loop,
    run_multi_head_learning_loop_batched,
)


class _HostileRepr:
    def __repr__(self) -> str:
        raise AssertionError("repr hook must not run")


class _HostileTuple(tuple[object, ...]):
    def __len__(self) -> int:
        raise AssertionError("tuple subclass hook must not run")


class _HostileDict(dict[str, object]):
    def __iter__(self):
        raise AssertionError("dict subclass hook must not run")

# =============================================================================
# Init tests
# =============================================================================


class TestMultiHeadInit:
    """Tests for MultiHeadMLPLearner.init."""

    @pytest.mark.parametrize("n_heads", [True, 0, 1.5, 2**31])
    def test_rejects_noncanonical_head_counts(self, n_heads: object):
        with pytest.raises(ValueError, match="n_heads"):
            MultiHeadMLPLearner(n_heads=n_heads)  # type: ignore[arg-type]

    @pytest.mark.parametrize("hidden_sizes", [[8], (True,), (1.5,), (0,), (2**31,)])
    def test_rejects_noncanonical_hidden_sizes(self, hidden_sizes: object):
        with pytest.raises(ValueError, match="hidden_sizes"):
            MultiHeadMLPLearner(
                n_heads=1,
                hidden_sizes=hidden_sizes,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "per_head_decay",
        [[0.5], (), (True,), (float("nan"),), (-0.1,), (1.1,)],
    )
    def test_rejects_invalid_per_head_trace_decay(self, per_head_decay: object):
        with pytest.raises(ValueError, match="per_head_gamma_lamda"):
            MultiHeadMLPLearner(
                n_heads=1,
                per_head_gamma_lamda=per_head_decay,  # type: ignore[arg-type]
            )

    def test_rejects_optimizers_without_shape_generic_hooks_at_construction(self):
        with pytest.raises(ValueError, match="optimizer.*MLP"):
            MultiHeadMLPLearner(n_heads=2, optimizer=ObGD())

        with pytest.raises(ValueError, match="head_optimizer.*MLP"):
            MultiHeadMLPLearner(n_heads=2, head_optimizer=AutostepGTDLambda())

    @pytest.mark.parametrize(
        "value",
        [True, np.bool_(True), 1.5, "2", 0, -1, 2**31],
    )
    def test_rejects_invalid_head_count_before_allocation(self, value: object):
        with pytest.raises(ValueError, match="n_heads"):
            MultiHeadMLPLearner(n_heads=value)  # type: ignore[arg-type]

    def test_rejected_integer_spoof_does_not_run_repr(self):
        with pytest.raises(ValueError, match="n_heads"):
            MultiHeadMLPLearner(n_heads=_HostileRepr())  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value",
        [True, np.bool_(True), 1.5, "2", 0, -1, 2**31],
    )
    def test_rejects_invalid_hidden_width_before_allocation(self, value: object):
        with pytest.raises(ValueError, match=r"hidden_sizes\[0\]"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(value,))  # type: ignore[arg-type]

    def test_direct_sequence_boundaries_require_actual_tuples(self):
        with pytest.raises(ValueError, match="hidden_sizes.*tuple"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=[2])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="per_head_gamma_lamda.*tuple"):
            MultiHeadMLPLearner(
                n_heads=1,
                hidden_sizes=(),
                per_head_gamma_lamda=[0.5],  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="per_head_gamma_lamda.*tuple"):
            MultiHeadMLPLearner(
                n_heads=1,
                hidden_sizes=(),
                per_head_gamma_lamda=_HostileTuple((0.5,)),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "scalar_type",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.longlong,
            np.ulonglong,
        ],
    )
    def test_numpy_integer_family_is_canonicalized(self, scalar_type: type):
        learner = MultiHeadMLPLearner(
            n_heads=scalar_type(2),  # type: ignore[call-arg,arg-type]
            hidden_sizes=(scalar_type(3),),  # type: ignore[call-arg,arg-type]
        )
        config = learner.to_config()
        assert type(learner.n_heads) is int
        assert type(learner.hidden_sizes[0]) is int
        assert type(config["n_heads"]) is int
        assert type(config["hidden_sizes"][0]) is int

    @pytest.mark.parametrize("feature_dim", [True, np.bool_(True), 1.5, "2", 0, -1, 2**31])
    def test_init_rejects_invalid_feature_dimension_before_allocation(
        self, feature_dim: object
    ):
        learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=())
        with pytest.raises(ValueError, match="feature_dim"):
            learner.init(feature_dim, jr.key(0))  # type: ignore[arg-type]

    def test_from_config_requires_json_lists_and_delegates_element_validation(self):
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(),
            per_head_gamma_lamda=(0.0, 0.5),
        )
        hidden_tuple = learner.to_config()
        hidden_tuple["hidden_sizes"] = ()
        with pytest.raises(ValueError, match="hidden_sizes.*list"):
            MultiHeadMLPLearner.from_config(hidden_tuple)

        per_head_tuple = learner.to_config()
        per_head_tuple["per_head_gamma_lamda"] = (0.0, 0.5)
        with pytest.raises(ValueError, match="per_head_gamma_lamda.*list"):
            MultiHeadMLPLearner.from_config(per_head_tuple)

        invalid_element = learner.to_config()
        invalid_element["per_head_gamma_lamda"] = [0.0, True]
        with pytest.raises(ValueError, match=r"per_head_gamma_lamda\[1\]"):
            MultiHeadMLPLearner.from_config(invalid_element)

    def test_from_config_requires_exact_outer_schema_before_dispatch(self):
        config = MultiHeadMLPLearner(n_heads=1, hidden_sizes=()).to_config()
        with pytest.raises(ValueError, match="actual dict"):
            MultiHeadMLPLearner.from_config(_HostileDict(config))  # type: ignore[arg-type]

        for field, value in (
            ("type", "WrongLearner"),
            ("type", _HostileRepr()),
            ("state_schema", "wrong-schema"),
            ("state_schema", _HostileRepr()),
        ):
            invalid = dict(config)
            invalid[field] = value
            with pytest.raises(ValueError):
                MultiHeadMLPLearner.from_config(invalid)

        missing = dict(config)
        missing.pop("optimizer")
        with pytest.raises(ValueError, match="fields"):
            MultiHeadMLPLearner.from_config(missing)
        unknown = dict(config)
        unknown["unknown"] = 1
        with pytest.raises(ValueError, match="fields"):
            MultiHeadMLPLearner.from_config(unknown)

    def test_constructor_rejects_derived_allocation_counts(self):
        with pytest.raises(ValueError, match="hidden_layer"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(46_341, 46_341))
        with pytest.raises(ValueError, match="head_weight"):
            MultiHeadMLPLearner(n_heads=46_341, hidden_sizes=(46_341,))
        with pytest.raises(ValueError, match="per_head_metrics"):
            MultiHeadMLPLearner(n_heads=(2**31 - 1) // 3 + 1, hidden_sizes=())

        with pytest.raises(ValueError, match="parameter_count"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(1, 2**31 - 1))

        boundary = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(1, 1_000))
        assert boundary.hidden_sizes == (1, 1_000)

    def test_init_preflights_derived_allocations_before_sparse_init(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls = 0

        def forbidden_allocator(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("allocator reached")

        monkeypatch.setattr(
            "alberta_framework.core.multi_head_learner.sparse_init",
            forbidden_allocator,
        )
        trunk = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(2,))
        with pytest.raises(ValueError, match="input_layer"):
            trunk.init(2**30, jr.key(0))
        linear = MultiHeadMLPLearner(n_heads=2, hidden_sizes=())
        with pytest.raises(ValueError, match="linear_head"):
            linear.init(2**30, jr.key(0))
        assert calls == 0

        boundary = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(1,))
        with pytest.raises(AssertionError, match="allocator reached"):
            boundary.init(268_435_450, jr.key(0))
        assert calls == 1

    def test_aggregate_direct_state_resources_fail_before_allocation(self):
        with pytest.raises(ValueError, match="direct_state_bytes"):
            MultiHeadMLPLearner(n_heads=134_217_728, hidden_sizes=())

        learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=())
        with pytest.raises(ValueError, match="direct_state_bytes"):
            learner.init(feature_dim=300_000_000, key=jr.key(0))

    def test_trunk_shapes_single_hidden(self):
        """Trunk with one hidden layer has correct shapes."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(32,), sparsity=0.0
        )
        state = learner.init(feature_dim=10, key=jr.key(42))

        # Trunk: 10 -> 32
        assert len(state.trunk_params.weights) == 1
        chex.assert_shape(state.trunk_params.weights[0], (32, 10))
        chex.assert_shape(state.trunk_params.biases[0], (32,))

    def test_trunk_shapes_two_hidden(self):
        """Trunk with two hidden layers has correct shapes."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(64, 32), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        assert len(state.trunk_params.weights) == 2
        chex.assert_shape(state.trunk_params.weights[0], (64, 5))
        chex.assert_shape(state.trunk_params.biases[0], (64,))
        chex.assert_shape(state.trunk_params.weights[1], (32, 64))
        chex.assert_shape(state.trunk_params.biases[1], (32,))

    def test_head_shapes(self):
        """Each head has a (1, H_last) weight and (1,) bias."""
        learner = MultiHeadMLPLearner(
            n_heads=4, hidden_sizes=(64, 32), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        assert len(state.head_params.weights) == 4
        assert len(state.head_params.biases) == 4
        for i in range(4):
            chex.assert_shape(state.head_params.weights[i], (1, 32))
            chex.assert_shape(state.head_params.biases[i], (1,))

    def test_traces_initialized_to_zero(self):
        """All trunk and head traces should be zero."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        for trace in state.trunk_traces:
            chex.assert_trees_all_close(trace, jnp.zeros_like(trace))

        for w_trace, b_trace in state.head_traces:
            chex.assert_trees_all_close(w_trace, jnp.zeros_like(w_trace))
            chex.assert_trees_all_close(b_trace, jnp.zeros_like(b_trace))

    def test_biases_initialized_to_zero(self):
        """All trunk and head biases should be zero."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        for bias in state.trunk_params.biases:
            chex.assert_trees_all_close(bias, jnp.zeros_like(bias))

        for bias in state.head_params.biases:
            chex.assert_trees_all_close(bias, jnp.zeros_like(bias))

    def test_sparsity_applied(self):
        """Trunk and head weights should be sparse when sparsity > 0."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(32,), sparsity=0.9
        )
        state = learner.init(feature_dim=10, key=jr.key(42))

        # Trunk layer: expect ~90% sparse
        zeros = jnp.sum(state.trunk_params.weights[0] == 0)
        total = state.trunk_params.weights[0].size
        sparsity = float(zeros) / total
        assert sparsity > 0.85

    def test_step_count_starts_at_zero(self):
        """step_count should be 0 after init."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))
        assert int(state.step_count) == 0

    def test_normalizer_state_init(self):
        """Normalizer state should be created when normalizer is provided."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            normalizer=EMANormalizer(),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        assert state.normalizer_state is not None
        chex.assert_shape(state.normalizer_state.mean, (5,))
        chex.assert_shape(state.normalizer_state.var, (5,))


# =============================================================================
# Predict tests
# =============================================================================


class TestMultiHeadPredict:
    """Tests for MultiHeadMLPLearner.predict."""

    def test_returns_n_heads_scalars(self):
        """predict should return array of shape (n_heads,)."""
        learner = MultiHeadMLPLearner(
            n_heads=4, hidden_sizes=(16,), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        preds = learner.predict(state, obs)

        chex.assert_shape(preds, (4,))
        chex.assert_tree_all_finite(preds)

    def test_deterministic(self):
        """Same state and observation should give same predictions."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])
        preds1 = learner.predict(state, obs)
        preds2 = learner.predict(state, obs)

        chex.assert_trees_all_close(preds1, preds2)


# =============================================================================
# Update tests — all heads active
# =============================================================================


class TestMultiHeadUpdateAllActive:
    """Tests for update with all heads active."""

    def test_correct_result_types(self):
        """Update should return MultiHeadMLPUpdateResult."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0, 3.0])

        result = learner.update(state, obs, targets)
        assert isinstance(result, MultiHeadMLPUpdateResult)
        assert isinstance(result.state, MultiHeadMLPState)

    def test_correct_shapes(self):
        """Metrics, predictions, errors should have correct shapes."""
        n_heads = 4
        learner = MultiHeadMLPLearner(
            n_heads=n_heads, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0, 3.0, 4.0])

        result = learner.update(state, obs, targets)

        chex.assert_shape(result.predictions, (n_heads,))
        chex.assert_shape(result.errors, (n_heads,))
        chex.assert_shape(result.per_head_metrics, (n_heads, 3))
        chex.assert_shape(result.trunk_bounding_metric, ())

    def test_no_nan_when_all_active(self):
        """All metrics should be finite when all heads are active."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0, 3.0])

        result = learner.update(state, obs, targets)

        chex.assert_tree_all_finite(result.predictions)
        chex.assert_tree_all_finite(result.errors)
        chex.assert_tree_all_finite(result.per_head_metrics)

    def test_error_reduction(self):
        """Multiple updates should reduce error on a fixed target."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), step_size=0.1, sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])
        targets = jnp.array([2.0, -1.0])

        initial_preds = learner.predict(state, obs)
        initial_se = float(jnp.sum((initial_preds - targets) ** 2))

        for _ in range(100):
            result = learner.update(state, obs, targets)
            state = result.state

        final_preds = learner.predict(state, obs)
        final_se = float(jnp.sum((final_preds - targets) ** 2))

        assert final_se < initial_se

    def test_step_count_increments(self):
        """step_count should increment by 1 each update."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        assert int(result.state.step_count) == 1

        result = learner.update(result.state, obs, targets)
        assert int(result.state.step_count) == 2


# =============================================================================
# Update tests — validation
# =============================================================================


class TestMultiHeadConstructorValidation:
    """Constructor scalars that reach init/to_config must be finite identities."""

    def test_rejects_wrong_length_per_head_gamma_lamda(self):
        """The tuple must have exactly one value per head."""
        with pytest.raises(ValueError, match="length n_heads"):
            MultiHeadMLPLearner(
                n_heads=2,
                hidden_sizes=(),
                sparsity=0.0,
                step_size=0.01,
                per_head_gamma_lamda=(0.5,),
            )

    @pytest.mark.parametrize("gl", [float("nan"), float("inf"), -0.1, 1.5, True, "0.5"])
    def test_rejects_non_finite_or_out_of_range_values(self, gl):
        """Each per-head trace decay must be a finite real in [0, 1]."""
        with pytest.raises(ValueError, match=r"per_head_gamma_lamda\[1\]"):
            MultiHeadMLPLearner(
                n_heads=2,
                hidden_sizes=(),
                sparsity=0.0,
                step_size=0.01,
                per_head_gamma_lamda=(0.5, gl),  # type: ignore[arg-type]
            )

    def test_rejects_class_spoof_without_invoking_float_hook(self):
        class Spoof:
            @property
            def __class__(self) -> type[float]:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                raise RuntimeError("must not run")

        with pytest.raises(ValueError, match=r"per_head_gamma_lamda\[1\]"):
            MultiHeadMLPLearner(
                n_heads=2,
                hidden_sizes=(),
                sparsity=0.0,
                step_size=0.01,
                per_head_gamma_lamda=(0.5, Spoof()),  # type: ignore[arg-type]
            )

    def test_canonicalizes_numpy_per_head_values(self):
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(),
            sparsity=0.0,
            step_size=0.01,
            per_head_gamma_lamda=(np.float32(0.25), np.float64(0.5)),
        )

        values = learner.to_config()["per_head_gamma_lamda"]
        assert all(type(value) is float for value in values)

    def test_legal_sparsity_and_slope_defaults_stay_bit_identical(self):
        learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=())
        assert learner._sparsity == 0.9
        assert type(learner._sparsity) is float
        assert learner._leaky_relu_slope == 0.01
        assert type(learner._leaky_relu_slope) is float
        assert learner._gamma == 0.0
        assert learner._lamda == 0.0
        assert learner._use_layer_norm is True
        assert learner._utility_decay == 0.99
        assert type(learner._utility_decay) is float

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("sparsity", float("nan")),
            ("sparsity", float("inf")),
            ("sparsity", True),
            ("sparsity", -0.1),
            ("sparsity", 1.5),
            ("leaky_relu_slope", float("nan")),
            ("leaky_relu_slope", True),
            ("leaky_relu_slope", -0.01),
            ("gamma", float("nan")),
            ("gamma", True),
            ("gamma", -0.1),
            ("lamda", float("nan")),
            ("lamda", True),
            ("lamda", 1.5),
            ("utility_decay", float("nan")),
            ("utility_decay", True),
            ("utility_decay", 1.0),
        ],
    )
    def test_rejects_non_finite_bool_and_out_of_range_constructor_scalars(
        self, field: str, value: object
    ):
        kwargs: dict[str, object] = {
            "n_heads": 1,
            "hidden_sizes": (),
            "sparsity": 0.0,
            "step_size": 0.01,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            MultiHeadMLPLearner(**kwargs)  # type: ignore[arg-type]

    def test_true_sparsity_does_not_serialize_as_full_mask(self):
        with pytest.raises(ValueError, match="sparsity"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), sparsity=True)

    def test_true_gamma_does_not_store_as_undiscounted_one(self):
        with pytest.raises(ValueError, match="gamma"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), gamma=True, lamda=0.0)

    def test_use_layer_norm_requires_exact_bool(self):
        with pytest.raises(ValueError, match="use_layer_norm"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), use_layer_norm=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="use_layer_norm"):
            MultiHeadMLPLearner(
                n_heads=1, hidden_sizes=(), use_layer_norm="true"  # type: ignore[arg-type]
            )

    def test_trace_mode_requires_the_enum(self):
        with pytest.raises(ValueError, match="trace_mode"):
            MultiHeadMLPLearner(
                n_heads=1,
                hidden_sizes=(),
                trace_mode="accumulating",  # type: ignore[arg-type]
            )

    def test_from_config_rejects_nan_sparsity_and_bool_gamma(self):
        config = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), sparsity=0.0).to_config()
        poisoned = dict(config)
        poisoned["sparsity"] = float("nan")
        with pytest.raises(ValueError, match="sparsity"):
            MultiHeadMLPLearner.from_config(poisoned)
        poisoned = dict(config)
        poisoned["gamma"] = True
        with pytest.raises(ValueError, match="gamma"):
            MultiHeadMLPLearner.from_config(poisoned)

    def test_rejects_class_spoofed_sparsity_without_invoking_float(self):
        class Spoof:
            @property
            def __class__(self) -> type[float]:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                raise RuntimeError("must not run")

        with pytest.raises(ValueError, match="sparsity"):
            MultiHeadMLPLearner(
                n_heads=1,
                hidden_sizes=(),
                sparsity=Spoof(),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sparsity": 2.0**-150},
            {"leaky_relu_slope": 5e-324},
            {"gamma": 2.0**-150},
            {"lamda": 2.0**-150},
            {"utility_decay": 5e-324},
            {"per_head_gamma_lamda": (2.0**-150,)},
        ],
    )
    def test_rejects_nonzero_float32_underflow(self, kwargs):
        with pytest.raises(ValueError, match="must remain nonzero"):
            MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), **kwargs)

    def test_rejects_trace_product_that_underflows_float32(self):
        with pytest.raises(ValueError, match=r"gamma \* lamda must remain nonzero"):
            MultiHeadMLPLearner(
                n_heads=1,
                hidden_sizes=(),
                gamma=2.0**-100,
                lamda=2.0**-100,
            )

    def test_nonnegative_float32_sinks_preserve_zero_and_minsubnormal(self):
        smallest_float32 = 2.0**-149
        learner = MultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=(),
            sparsity=smallest_float32,
            leaky_relu_slope=0.0,
            gamma=0.0,
            lamda=smallest_float32,
            utility_decay=0.0,
            per_head_gamma_lamda=(smallest_float32,),
        )
        config = learner.to_config()
        assert config["sparsity"] == smallest_float32
        assert config["leaky_relu_slope"] == 0.0
        assert config["lamda"] == smallest_float32
        assert config["utility_decay"] == 0.0
        assert config["per_head_gamma_lamda"] == [smallest_float32]


class TestMultiHeadUpdateValidation:
    """``update`` must reject targets that are not one value per head.

    JAX clamps static out-of-bounds indices instead of raising, so a
    shorter-than-``n_heads`` targets array (a scalar or length-1 array is the
    most common accident) silently reused its single value for every head:
    every head reported a finite squared error and trained, instead of the
    NaN/inactive behavior the caller intended for the heads it never supplied.
    """

    @pytest.mark.parametrize("shape", [(1,), (), (4, 1), (5,)])
    def test_update_rejects_targets_that_are_not_one_per_head(self, shape):
        learner = MultiHeadMLPLearner(
            n_heads=4, hidden_sizes=(8,), sparsity=0.0, step_size=0.01,
        )
        state = learner.init(feature_dim=3, key=jr.key(0))
        obs = jnp.array([1.0, -0.5, 0.25])
        with pytest.raises(ValueError, match=r"targets must have shape \(4,\)"):
            learner.update(state, obs, jnp.full(shape, 2.0, dtype=jnp.float32))

    def test_length_one_targets_previously_silently_trained_every_head(self):
        """Regression guard: shape (1,) must not longer look like all-active.

        Before the shape check, a length-1 ``targets`` array silently
        clamped-indexed the same value onto every head, producing a finite
        (non-NaN) error for every head instead of raising. This test would
        have failed to reject the mismatch on the pre-fix implementation
        (the ``update`` call would have returned a result with no NaNs
        instead of raising).
        """
        learner = MultiHeadMLPLearner(
            n_heads=4, hidden_sizes=(8,), sparsity=0.0, step_size=0.01,
        )
        state = learner.init(feature_dim=3, key=jr.key(0))
        obs = jnp.array([1.0, -0.5, 0.25])
        with pytest.raises(ValueError, match=r"targets must have shape \(4,\)"):
            learner.update(state, obs, jnp.array([2.0], dtype=jnp.float32))


# =============================================================================
# Update tests — partial active
# =============================================================================


class TestMultiHeadUpdatePartialActive:
    """Tests for update with some heads inactive (NaN targets)."""

    def test_nan_metrics_for_inactive(self):
        """Inactive heads should have NaN in errors and metrics."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, jnp.nan, 3.0])  # Head 1 inactive

        result = learner.update(state, obs, targets)

        # Head 0 and 2 should be finite
        assert jnp.isfinite(result.errors[0])
        assert jnp.isfinite(result.errors[2])

        # Head 1 should be NaN
        assert jnp.isnan(result.errors[1])
        assert jnp.all(jnp.isnan(result.per_head_metrics[1]))

    def test_inactive_head_params_unchanged(self):
        """Inactive head params should not change."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, jnp.nan, 3.0])  # Head 1 inactive

        result = learner.update(state, obs, targets)

        # Head 1 weights/biases should be unchanged
        chex.assert_trees_all_close(
            result.state.head_params.weights[1],
            state.head_params.weights[1],
        )
        chex.assert_trees_all_close(
            result.state.head_params.biases[1],
            state.head_params.biases[1],
        )

        # Active heads should have changed
        assert not jnp.allclose(
            result.state.head_params.weights[0],
            state.head_params.weights[0],
        )

    def test_predictions_always_computed(self):
        """Predictions should be computed for all heads, even inactive ones."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([jnp.nan, jnp.nan, 1.0])

        result = learner.update(state, obs, targets)

        # All predictions should be finite
        chex.assert_tree_all_finite(result.predictions)


# =============================================================================
# Update tests — no heads active
# =============================================================================


class TestMultiHeadUpdateNoneActive:
    """Tests for update with no heads active (all NaN targets)."""

    def test_head_params_unchanged(self):
        """All head params should remain unchanged when no heads are active."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([jnp.nan, jnp.nan])

        result = learner.update(state, obs, targets)

        for i in range(2):
            chex.assert_trees_all_close(
                result.state.head_params.weights[i],
                state.head_params.weights[i],
            )
            chex.assert_trees_all_close(
                result.state.head_params.biases[i],
                state.head_params.biases[i],
            )

    def test_normalizer_still_updates(self):
        """Normalizer should update even when no heads are active."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            normalizer=EMANormalizer(),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = jnp.array([jnp.nan, jnp.nan])

        result = learner.update(state, obs, targets)

        # Normalizer mean should have changed
        assert not jnp.allclose(
            result.state.normalizer_state.mean,
            state.normalizer_state.mean,
        )

    def test_step_count_still_increments(self):
        """step_count should still increment even with no active heads."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        targets = jnp.array([jnp.nan, jnp.nan])
        result = learner.update(state, jnp.ones(5), targets)
        assert int(result.state.step_count) == 1


# =============================================================================
# Composition tests
# =============================================================================


class TestMultiHeadComposition:
    """Tests for composing with different optimizers/bounders/normalizers."""

    def test_with_obgd_bounding(self):
        """Should work with ObGDBounding."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        chex.assert_tree_all_finite(result.predictions)
        chex.assert_tree_all_finite(result.per_head_metrics)

    def test_with_agc_bounding(self):
        """Should work with AGCBounding."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            bounder=AGCBounding(clip_factor=0.01),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        chex.assert_tree_all_finite(result.predictions)

    def test_with_ema_normalizer(self):
        """Should work with EMANormalizer."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            normalizer=EMANormalizer(decay=0.95),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        chex.assert_tree_all_finite(result.predictions)
        assert result.state.normalizer_state is not None

    def test_with_welford_normalizer(self):
        """Should work with WelfordNormalizer."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            normalizer=WelfordNormalizer(),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        chex.assert_tree_all_finite(result.predictions)

    def test_with_autostep_optimizer(self):
        """Should work with Autostep optimizer."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        chex.assert_tree_all_finite(result.predictions)
        chex.assert_tree_all_finite(result.per_head_metrics)


# =============================================================================
# Gradient correctness
# =============================================================================


class TestMultiHeadGradientCorrectness:
    """Tests verifying VJP trunk gradients match N separate jax.grad calls."""

    def test_vjp_matches_separate_grads(self):
        """Accumulated VJP cotangent should match sum of per-head grads."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0, 3.0])

        slope = learner._leaky_relu_slope
        ln = learner._use_layer_norm

        # Compute via N separate jax.grad calls
        accumulated_w_grads = [
            jnp.zeros_like(w) for w in state.trunk_params.weights
        ]
        accumulated_b_grads = [
            jnp.zeros_like(b) for b in state.trunk_params.biases
        ]

        for i in range(3):
            # Full forward: trunk + head_i
            def full_forward_i(
                trunk_w: tuple, trunk_b: tuple, head_idx: int = i
            ) -> jax.Array:
                hidden = MultiHeadMLPLearner._trunk_forward(
                    trunk_w, trunk_b, obs, slope, ln
                )
                return MultiHeadMLPLearner._head_forward(
                    state.head_params.weights[head_idx],
                    state.head_params.biases[head_idx],
                    hidden,
                )

            w_grads, b_grads = jax.grad(full_forward_i, argnums=(0, 1))(
                state.trunk_params.weights, state.trunk_params.biases
            )

            error_i = targets[i] - full_forward_i(
                state.trunk_params.weights, state.trunk_params.biases
            )

            for j in range(len(accumulated_w_grads)):
                accumulated_w_grads[j] = (
                    accumulated_w_grads[j] + error_i * w_grads[j]
                )
                accumulated_b_grads[j] = (
                    accumulated_b_grads[j] + error_i * b_grads[j]
                )

        # Compute via VJP (as the learner does)
        def trunk_fn(weights, biases):
            return MultiHeadMLPLearner._trunk_forward(
                weights, biases, obs, slope, ln
            )

        hidden, trunk_vjp_fn = jax.vjp(
            trunk_fn,
            state.trunk_params.weights,
            state.trunk_params.biases,
        )

        # Build cotangent
        h_last = hidden.shape[0]
        cotangent = jnp.zeros(h_last, dtype=jnp.float32)
        for i in range(3):
            pred_i = MultiHeadMLPLearner._head_forward(
                state.head_params.weights[i],
                state.head_params.biases[i],
                hidden,
            )
            error_i = targets[i] - pred_i
            cotangent = cotangent + error_i * jnp.squeeze(
                state.head_params.weights[i]
            )

        vjp_w_grads, vjp_b_grads = trunk_vjp_fn(cotangent)

        # Compare
        for j in range(len(accumulated_w_grads)):
            chex.assert_trees_all_close(
                vjp_w_grads[j], accumulated_w_grads[j], atol=1e-5
            )
            chex.assert_trees_all_close(
                vjp_b_grads[j], accumulated_b_grads[j], atol=1e-5
            )


# =============================================================================
# Metrics utility
# =============================================================================


class TestMultiHeadMetricsToDicts:
    """Tests for multi_head_metrics_to_dicts."""

    def test_all_active(self):
        """All active heads should produce dicts."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0, 3.0])

        result = learner.update(state, obs, targets)
        dicts = multi_head_metrics_to_dicts(result)

        assert len(dicts) == 3
        for d in dicts:
            assert d is not None
            assert "squared_error" in d
            assert "error" in d
            assert "mean_step_size" in d

    def test_partial_active(self):
        """Inactive heads should produce None."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, jnp.nan, 3.0])

        result = learner.update(state, obs, targets)
        dicts = multi_head_metrics_to_dicts(result)

        assert dicts[0] is not None
        assert dicts[1] is None
        assert dicts[2] is not None

    def test_none_active(self):
        """All NaN targets should produce all None."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([jnp.nan, jnp.nan])

        result = learner.update(state, obs, targets)
        dicts = multi_head_metrics_to_dicts(result)

        assert dicts[0] is None
        assert dicts[1] is None


# =============================================================================
# Scan loop tests
# =============================================================================


class TestRunMultiHeadLearningLoop:
    """Tests for run_multi_head_learning_loop."""

    def test_correct_shapes(self):
        """Scan loop should return correct metric shapes."""
        n_heads = 3
        num_steps = 50
        feature_dim = 5

        learner = MultiHeadMLPLearner(
            n_heads=n_heads, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=feature_dim, key=jr.key(0))

        # Generate synthetic data
        key = jr.key(42)
        k1, k2 = jr.split(key)
        observations = jr.normal(k1, (num_steps, feature_dim))
        targets = jr.normal(k2, (num_steps, n_heads))

        result = run_multi_head_learning_loop(
            learner, state, observations, targets
        )

        assert isinstance(result, MultiHeadLearningResult)
        chex.assert_shape(
            result.per_head_metrics, (num_steps, n_heads, 3)
        )

    def test_deterministic(self):
        """Same inputs should give identical results."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(0))

        key = jr.key(42)
        k1, k2 = jr.split(key)
        observations = jr.normal(k1, (30, 5))
        targets = jr.normal(k2, (30, 2))

        result1 = run_multi_head_learning_loop(
            learner, state, observations, targets
        )
        result2 = run_multi_head_learning_loop(
            learner, state, observations, targets
        )

        chex.assert_trees_all_close(
            result1.per_head_metrics, result2.per_head_metrics
        )

    def test_nan_target_handling(self):
        """Should handle NaN targets correctly in scan loop."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(0))

        observations = jr.normal(jr.key(42), (20, 5))
        # Head 0 always active, head 1 active only first 10 steps
        targets = jr.normal(jr.key(99), (20, 2))
        targets = targets.at[10:, 1].set(jnp.nan)

        result = run_multi_head_learning_loop(
            learner, state, observations, targets
        )

        # Head 0 metrics should all be finite
        assert jnp.all(jnp.isfinite(result.per_head_metrics[:, 0, :]))

        # Head 1 metrics: first 10 steps finite, last 10 NaN
        assert jnp.all(jnp.isfinite(result.per_head_metrics[:10, 1, :]))
        assert jnp.all(jnp.isnan(result.per_head_metrics[10:, 1, 0]))

    def test_with_normalizer(self):
        """Should work with normalizer in scan loop."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            normalizer=EMANormalizer(),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(0))

        observations = jr.normal(jr.key(42), (30, 5))
        targets = jr.normal(jr.key(99), (30, 2))

        result = run_multi_head_learning_loop(
            learner, state, observations, targets
        )

        chex.assert_shape(result.per_head_metrics, (30, 2, 3))
        # Normalizer should have updated
        assert result.state.normalizer_state is not None


# =============================================================================
# Batched loop tests
# =============================================================================


class TestRunMultiHeadLearningLoopBatched:
    """Tests for run_multi_head_learning_loop_batched."""

    def test_correct_shapes(self):
        """Batched loop should return correctly shaped results."""
        n_heads = 3
        num_steps = 30
        feature_dim = 5
        n_seeds = 4

        learner = MultiHeadMLPLearner(
            n_heads=n_heads, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )

        key = jr.key(42)
        k1, k2, k3 = jr.split(key, 3)
        observations = jr.normal(k1, (num_steps, feature_dim))
        targets = jr.normal(k2, (num_steps, n_heads))
        keys = jr.split(k3, n_seeds)

        result = run_multi_head_learning_loop_batched(
            learner, observations, targets, keys
        )

        assert isinstance(result, BatchedMultiHeadResult)
        chex.assert_shape(
            result.per_head_metrics, (n_seeds, num_steps, n_heads, 3)
        )

    def test_matches_sequential(self):
        """Batched results should match sequential for each seed."""
        n_heads = 2
        num_steps = 20
        feature_dim = 5
        n_seeds = 3

        learner = MultiHeadMLPLearner(
            n_heads=n_heads, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )

        key = jr.key(42)
        k1, k2, k3 = jr.split(key, 3)
        observations = jr.normal(k1, (num_steps, feature_dim))
        targets = jr.normal(k2, (num_steps, n_heads))
        keys = jr.split(k3, n_seeds)

        # Batched
        batched_result = run_multi_head_learning_loop_batched(
            learner, observations, targets, keys
        )

        # Sequential
        for i in range(n_seeds):
            state_i = learner.init(feature_dim, keys[i])
            seq_result = run_multi_head_learning_loop(
                learner, state_i, observations, targets
            )
            chex.assert_trees_all_close(
                batched_result.per_head_metrics[i],
                seq_result.per_head_metrics,
                rtol=1e-4,
            )

    def test_different_seeds_different_results(self):
        """Different seeds should produce different metrics."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )

        key = jr.key(42)
        k1, k2, k3 = jr.split(key, 3)
        observations = jr.normal(k1, (30, 5))
        targets = jr.normal(k2, (30, 2))
        keys = jr.split(k3, 3)

        result = run_multi_head_learning_loop_batched(
            learner, observations, targets, keys
        )

        # Different seeds should give different final metrics
        assert not jnp.allclose(
            result.per_head_metrics[0], result.per_head_metrics[1]
        )


class TestMultiHeadLifecycleTracking:
    """Tests for multi-head MLP lifecycle tracking (birth_timestamp, uptime_s)."""

    def test_birth_timestamp_set(self):
        """birth_timestamp should be set at init."""
        before = time.time()
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))
        after = time.time()
        assert before <= state.birth_timestamp <= after

    def test_birth_timestamp_survives_update(self):
        """birth_timestamp should not change across updates."""
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))
        original_ts = state.birth_timestamp

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])
        result = learner.update(state, obs, targets)
        assert result.state.birth_timestamp == original_ts

    def test_uptime_starts_at_zero(self):
        """uptime_s should be 0.0 after init."""
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))
        assert state.uptime_s == 0.0

    def test_uptime_increases_after_loop(self):
        """uptime_s should be > 0 after run_multi_head_learning_loop."""
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(16,), sparsity=0.0)
        key = jr.key(42)
        k1, k2, k3 = jr.split(key, 3)

        state = learner.init(feature_dim=5, key=k1)
        observations = jr.normal(k2, (50, 5))
        targets = jr.normal(k3, (50, 2))

        result = run_multi_head_learning_loop(learner, state, observations, targets)
        assert result.state.uptime_s > 0.0

    def test_uptime_accumulates(self):
        """uptime_s should accumulate across sequential loops."""
        learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(16,), sparsity=0.0)
        key = jr.key(42)
        k1, k2, k3, k4, k5 = jr.split(key, 5)

        state = learner.init(feature_dim=5, key=k1)
        obs1 = jr.normal(k2, (50, 5))
        tgt1 = jr.normal(k3, (50, 2))

        result1 = run_multi_head_learning_loop(learner, state, obs1, tgt1)
        uptime_after_first = result1.state.uptime_s
        assert uptime_after_first > 0.0

        obs2 = jr.normal(k4, (50, 5))
        tgt2 = jr.normal(k5, (50, 2))
        result2 = run_multi_head_learning_loop(
            learner, result1.state, obs2, tgt2
        )
        assert result2.state.uptime_s > uptime_after_first


# =============================================================================
# Hybrid optimizer tests
# =============================================================================


class TestMultiHeadHybridOptimizer:
    """Tests for MultiHeadMLPLearner with head_optimizer."""

    def test_hybrid_init_creates_different_states(self):
        """Head optimizer states should differ from trunk when using hybrid."""
        from alberta_framework.core.types import AutostepParamState, LMSState

        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        # Trunk optimizer states should be LMS
        for trunk_opt in state.trunk_optimizer_states:
            assert isinstance(trunk_opt, LMSState)

        # Head optimizer states should be Autostep
        for w_opt, b_opt in state.head_optimizer_states:
            assert isinstance(w_opt, AutostepParamState)
            assert isinstance(b_opt, AutostepParamState)

    def test_hybrid_update_runs(self):
        """Update with hybrid optimizer should work."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        assert isinstance(result, MultiHeadMLPUpdateResult)
        chex.assert_tree_all_finite(result.predictions)
        chex.assert_tree_all_finite(result.per_head_metrics)

    def test_hybrid_scan_loop(self):
        """Full scan loop with hybrid optimizer should work."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )

        key = jr.key(42)
        k1, k2, k3 = jr.split(key, 3)
        state = learner.init(feature_dim=5, key=k1)
        observations = jr.normal(k2, (50, 5))
        targets = jr.normal(k3, (50, 2))

        result = run_multi_head_learning_loop(
            learner, state, observations, targets
        )

        assert isinstance(result, MultiHeadLearningResult)
        chex.assert_shape(result.per_head_metrics, (50, 2, 3))

    def test_hybrid_default_none_matches_uniform(self):
        """head_optimizer=None should produce same results as explicit single optimizer."""
        key = jr.key(42)
        k1, k2, k3 = jr.split(key, 3)
        observations = jr.normal(k2, (30, 5))
        targets = jr.normal(k3, (30, 2))

        learner_default = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0, bounder=ObGDBounding(kappa=2.0),
        )
        learner_explicit = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0, head_optimizer=None,
            bounder=ObGDBounding(kappa=2.0),
        )

        state_default = learner_default.init(feature_dim=5, key=k1)
        state_explicit = learner_explicit.init(feature_dim=5, key=k1)

        result_default = run_multi_head_learning_loop(
            learner_default, state_default, observations, targets
        )
        result_explicit = run_multi_head_learning_loop(
            learner_explicit, state_explicit, observations, targets
        )

        chex.assert_trees_all_close(
            result_default.per_head_metrics,
            result_explicit.per_head_metrics,
        )


# =============================================================================
# Linear baseline tests (hidden_sizes=())
# =============================================================================


class TestMultiHeadLinearBaseline:
    """Tests for MultiHeadMLPLearner with hidden_sizes=() (linear model)."""

    def test_init_succeeds(self):
        """hidden_sizes=() should init without error."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        # No trunk layers
        assert len(state.trunk_params.weights) == 0
        assert len(state.trunk_params.biases) == 0
        assert len(state.trunk_traces) == 0
        assert len(state.trunk_optimizer_states) == 0

    def test_head_shapes_match_input(self):
        """Heads should project from feature_dim directly."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        assert len(state.head_params.weights) == 3
        for i in range(3):
            chex.assert_shape(state.head_params.weights[i], (1, 5))
            chex.assert_shape(state.head_params.biases[i], (1,))

    def test_predict_correct_shape(self):
        """predict should return (n_heads,) array."""
        learner = MultiHeadMLPLearner(
            n_heads=4, hidden_sizes=(), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        preds = learner.predict(state, obs)

        chex.assert_shape(preds, (4,))
        chex.assert_tree_all_finite(preds)

    def test_update_correct_shape(self):
        """update should return correct shapes."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0, 3.0])

        result = learner.update(state, obs, targets)

        chex.assert_shape(result.predictions, (3,))
        chex.assert_shape(result.errors, (3,))
        chex.assert_shape(result.per_head_metrics, (3, 3))
        chex.assert_tree_all_finite(result.predictions)
        chex.assert_tree_all_finite(result.per_head_metrics)

    def test_state_updates(self):
        """Head params should change after update."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), step_size=0.1, sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)

        # Head weights should have changed
        assert not jnp.allclose(
            result.state.head_params.weights[0],
            state.head_params.weights[0],
        )
        assert int(result.state.step_count) == 1

    def test_error_reduction(self):
        """Multiple updates on fixed target should reduce error."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), step_size=0.1, sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])
        targets = jnp.array([2.0, -1.0])

        initial_preds = learner.predict(state, obs)
        initial_se = float(jnp.sum((initial_preds - targets) ** 2))

        for _ in range(100):
            result = learner.update(state, obs, targets)
            state = result.state

        final_preds = learner.predict(state, obs)
        final_se = float(jnp.sum((final_preds - targets) ** 2))

        assert final_se < initial_se

    def test_nan_masking(self):
        """NaN targets should leave inactive heads unchanged."""
        learner = MultiHeadMLPLearner(
            n_heads=3, hidden_sizes=(), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, jnp.nan, 3.0])

        result = learner.update(state, obs, targets)

        # Head 1 should be unchanged
        chex.assert_trees_all_close(
            result.state.head_params.weights[1],
            state.head_params.weights[1],
        )
        assert jnp.isnan(result.errors[1])

    def test_scan_loop(self):
        """Should work in scan-based learning loop."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(0))

        key = jr.key(42)
        k1, k2 = jr.split(key)
        observations = jr.normal(k1, (30, 5))
        targets = jr.normal(k2, (30, 2))

        result = run_multi_head_learning_loop(
            learner, state, observations, targets
        )

        assert isinstance(result, MultiHeadLearningResult)
        chex.assert_shape(result.per_head_metrics, (30, 2, 3))

    def test_with_normalizer(self):
        """Should work with EMANormalizer."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), sparsity=0.0,
            normalizer=EMANormalizer(),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        targets = jnp.array([1.0, 2.0])

        result = learner.update(state, obs, targets)
        chex.assert_tree_all_finite(result.predictions)
        assert result.state.normalizer_state is not None

    def test_zero_trace_decay_does_not_multiply_inf_head_traces(self) -> None:
        """Default gamma*lamda is 0; 0 * inf head traces is NaN and would freeze."""
        learner = MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), sparsity=0.0, optimizer=LMS(0.1)
        )
        state = learner.init(feature_dim=3, key=jr.key(2))
        poisoned = [
            (jnp.full_like(w, jnp.inf), jnp.full_like(b, jnp.inf))
            for w, b in state.head_traces
        ]
        state = state.replace(head_traces=tuple(poisoned))
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = learner.update(
            state,
            jnp.ones(3, dtype=jnp.float32),
            jnp.array([1.0, 0.5], dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        for w_trace, b_trace in result.state.head_traces:
            assert bool(jnp.all(jnp.isfinite(w_trace)))
            assert bool(jnp.all(jnp.isfinite(b_trace)))

    def test_zero_trace_decay_recovers_inf_trunk_traces(self) -> None:
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(4,),
            sparsity=0.0,
            optimizer=LMS(0.1),
        )
        state = learner.init(feature_dim=3, key=jr.key(3))
        state = state.replace(
            trunk_traces=tuple(
                jnp.full_like(trace, jnp.inf) for trace in state.trunk_traces
            )
        )

        result = learner.update(
            state,
            jnp.ones(3, dtype=jnp.float32),
            jnp.array([1.0, 0.5], dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        for trace in result.state.trunk_traces:
            assert bool(jnp.all(jnp.isfinite(trace)))

    def test_replacing_zero_decay_skips_inf_trace_at_zero_gradient(self) -> None:
        learner = MultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=(),
            sparsity=0.0,
            optimizer=LMS(0.1),
            trace_mode=TraceMode.REPLACING,
        )
        state = learner.init(feature_dim=3, key=jr.key(4))
        old_w, old_b = state.head_traces[0]
        state = state.replace(
            head_traces=((jnp.full_like(old_w, jnp.inf), old_b),)
        )

        result = learner.update(
            state,
            jnp.zeros(3, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.state.head_traces[0][0])))

    def test_per_head_zero_decay_only_skips_disabled_head_trace(self) -> None:
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(),
            sparsity=0.0,
            optimizer=LMS(0.1),
            per_head_gamma_lamda=(0.0, 0.5),
        )
        state = learner.init(feature_dim=3, key=jr.key(5))
        first_w, first_b = state.head_traces[0]
        second_w, second_b = state.head_traces[1]
        state = state.replace(
            head_traces=(
                (jnp.full_like(first_w, jnp.inf), jnp.full_like(first_b, jnp.inf)),
                (jnp.ones_like(second_w), jnp.ones_like(second_b)),
            )
        )
        observation = jnp.ones(3, dtype=jnp.float32)

        result = learner.update(
            state,
            observation,
            jnp.array([1.0, 0.5], dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        first_result_w, first_result_b = result.state.head_traces[0]
        second_result_w, second_result_b = result.state.head_traces[1]
        chex.assert_trees_all_close(first_result_w, observation.reshape(1, -1))
        chex.assert_trees_all_close(first_result_b, jnp.ones_like(first_result_b))
        chex.assert_trees_all_close(
            second_result_w,
            0.5 * jnp.ones_like(second_w) + observation.reshape(1, -1),
        )
        chex.assert_trees_all_close(
            second_result_b,
            0.5 * jnp.ones_like(second_b) + jnp.ones_like(second_b),
        )

    def test_zero_utility_decay_recovers_inf_hidden_utilities(self) -> None:
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(4,),
            sparsity=0.0,
            optimizer=LMS(0.1),
            utility_decay=0.0,
        )
        state = learner.init(feature_dim=3, key=jr.key(6))
        state = state.replace(
            hidden_unit_utilities=tuple(
                jnp.full_like(utility, jnp.inf)
                for utility in state.hidden_unit_utilities
            )
        )

        result = learner.update(
            state,
            jnp.ones(3, dtype=jnp.float32),
            jnp.array([1.0, 0.5], dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        for utility in result.state.hidden_unit_utilities:
            assert bool(jnp.all(jnp.isfinite(utility)))
