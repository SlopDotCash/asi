"""Committed regressions for MultiHeadMLPLearner fail-closed validation."""

import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner

_INT32_MAX = 2**31 - 1


def test_accepts_numpy_int_types():
    for typ in (
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
    ):
        learner = MultiHeadMLPLearner(n_heads=typ(2), hidden_sizes=(typ(4),))
        assert learner.n_heads == 2
        state = learner.init(feature_dim=typ(3), key=jr.key(0))
        assert state is not None


def test_rejects_bool_and_np_bool():
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=True)
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=np.bool_(True))
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(True,))


def test_rejects_float_str_nan_inf():
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2.0)
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=float("nan"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=float("inf"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(1.5,))  # type: ignore[arg-type]


def test_rejects_hostile_subclass():
    class EvilInt(int):
        def __repr__(self):
            raise AssertionError("hostile repr must not be invoked")

    with pytest.raises(ValueError) as exc:
        MultiHeadMLPLearner(n_heads=EvilInt(2))
    assert "EvilInt" not in str(exc.value)

    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(EvilInt(4),))


def test_hostile_repr_not_interpolated_for_int():
    class Hostile:
        def __index__(self):
            return 2

        def __repr__(self):
            raise AssertionError("repr escape")

    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=Hostile())  # type: ignore[arg-type]


def test_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=0)
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=-1)
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=_INT32_MAX + 1)
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2**60)
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(0,))
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(2**60,))


def test_hidden_sizes_exact_tuple_guard():
    with pytest.raises(ValueError, match="hidden_sizes must be an actual tuple"):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=[4, 4])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_sizes must be an actual tuple"):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes="4")  # type: ignore[arg-type]
    # valid tuple passes
    learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(4, 4))
    assert learner._hidden_sizes == (4, 4)
    # linear baseline empty tuple preserved
    linear = MultiHeadMLPLearner(n_heads=2, hidden_sizes=())
    assert linear._hidden_sizes == ()
    state = linear.init(feature_dim=4, key=jr.key(0))
    assert state.trunk_params.weights == ()


def test_from_config_exact_list_guard():
    base = {
        "type": "MultiHeadMLPLearner",
        "n_heads": 2,
        "hidden_sizes": (4, 4),  # tuple not list -> should fail
        "optimizer": {"type": "LMS", "step_size": 1.0},
        "sparsity": 0.9,
        "leaky_relu_slope": 0.01,
        "use_layer_norm": True,
        "gamma": 0.0,
        "lamda": 0.0,
        "trace_mode": "accumulating",
        "utility_decay": 0.99,
    }
    with pytest.raises(ValueError, match="hidden_sizes must be a list"):
        MultiHeadMLPLearner.from_config(dict(base))
    # correct list passes
    base["hidden_sizes"] = [4, 4]
    learner = MultiHeadMLPLearner.from_config(dict(base))
    assert learner._hidden_sizes == (4, 4)


def test_derived_hidden_hidden_overflow():
    with pytest.raises(ValueError, match=r"hidden_sizes\[0\] \* hidden_sizes\[1\]"):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(50000, 50000))


def test_derived_n_heads_hidden_overflow():
    with pytest.raises(ValueError, match=r"n_heads \* hidden_sizes\[-1\]"):
        MultiHeadMLPLearner(n_heads=50000, hidden_sizes=(50000,))


def test_derived_feature_dim_hidden_overflow():
    learner = MultiHeadMLPLearner(n_heads=2, hidden_sizes=(50000,))
    with pytest.raises(ValueError, match=r"feature_dim \* hidden_sizes\[0\]"):
        learner.init(feature_dim=50000, key=jr.key(0))


def test_derived_feature_dim_n_heads_overflow_linear():
    learner = MultiHeadMLPLearner(n_heads=50000, hidden_sizes=())
    with pytest.raises(ValueError, match=r"feature_dim \* n_heads"):
        learner.init(feature_dim=50000, key=jr.key(0))


def test_valid_derived_within_bounds():
    learner = MultiHeadMLPLearner(n_heads=4, hidden_sizes=(8, 8))
    state = learner.init(feature_dim=8, key=jr.key(1))
    assert state is not None
    # linear baseline within bounds
    lin = MultiHeadMLPLearner(n_heads=4, hidden_sizes=())
    s2 = lin.init(feature_dim=8, key=jr.key(2))
    assert s2 is not None


def test_per_head_decay_validated():
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), per_head_gamma_lamda=(True, 0.5)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(
            n_heads=2, hidden_sizes=(), per_head_gamma_lamda=(float("nan"), 0.5)
        )
    with pytest.raises(ValueError):
        MultiHeadMLPLearner(n_heads=2, hidden_sizes=(), per_head_gamma_lamda=(1.5, 0.5))
    # from_config per_head must be list not tuple
    base = {
        "type": "MultiHeadMLPLearner",
        "n_heads": 2,
        "hidden_sizes": [],
        "optimizer": {"type": "LMS", "step_size": 1.0},
        "per_head_gamma_lamda": (0.5, 0.5),  # tuple not list
        "sparsity": 0.9,
        "leaky_relu_slope": 0.01,
        "use_layer_norm": True,
        "gamma": 0.0,
        "lamda": 0.0,
        "trace_mode": "accumulating",
        "utility_decay": 0.99,
    }
    with pytest.raises(ValueError, match="per_head_gamma_lamda must be a list"):
        MultiHeadMLPLearner.from_config(dict(base))
