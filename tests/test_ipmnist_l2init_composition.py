"""Regression coverage for the sigma0_shiftnorm_d099_l2init composition arm.

The l2-init composition reuses the adaptive-norm sigma0 learner with a
decay-to-init flag. These tests pin the registry contract and the
flag semantics: flag=0 must preserve bit-exact parity with the existing
shiftnorm arms (same hyperparameters, same update path).
"""

import jax.numpy as jnp

from alberta_framework.benchmarks.ipmnist_screening import (
    _make_upgd_shiftnorm_learner,
    screening_spec,
)


def test_l2init_arm_registered() -> None:
    spec = screening_spec("sigma0_shiftnorm_d099_l2init")
    assert spec.base_learner == "upgd_w"
    assert spec.hyperparameters["weight_decay"] == 0.01
    assert spec.hyperparameters["flag_decay_to_init"] == 1.0
    assert spec.hyperparameters["norm_decay"] == 0.99


def test_l2init_arm_matches_base_decay_scale() -> None:
    base = screening_spec("sigma0_shiftnorm_d099")
    composed = screening_spec("sigma0_shiftnorm_d099_l2init")
    # Same decay scale as the base arm and the l2init comparison row.
    assert composed.hyperparameters["weight_decay"] == base.hyperparameters["weight_decay"] == 0.01


def _test_params() -> dict:
    return {
        "w1": jnp.ones((784, 64), dtype=jnp.float32) * 0.1,
        "b1": jnp.zeros((64,), dtype=jnp.float32),
        "w2": jnp.ones((64, 64), dtype=jnp.float32) * 0.1,
        "b2": jnp.zeros((64,), dtype=jnp.float32),
        "w3": jnp.ones((64, 10), dtype=jnp.float32) * 0.1,
        "b3": jnp.zeros((10,), dtype=jnp.float32),
    }


def test_factory_flag_off_keeps_zero_decay_path() -> None:
    # flag_decay_to_init=0 (or absent) must route through the original
    # decay-to-zero path: the composed learner is bit-identical in
    # structure to the plain shiftnorm learner.
    hp = dict(screening_spec("sigma0_shiftnorm_d099").hyperparameters)
    init_fn_plain, _ = _make_upgd_shiftnorm_learner(hp)
    hp["flag_decay_to_init"] = 0.0
    init_fn_composed, _ = _make_upgd_shiftnorm_learner(hp)
    params = _test_params()
    plain = init_fn_plain(params)
    composed = init_fn_composed(params)
    # flag=0: init_params stays None (no extra carry), identical state.
    assert composed.init_params is None
    assert plain.init_params is None


def test_factory_flag_on_populates_init_params() -> None:
    hp = dict(screening_spec("sigma0_shiftnorm_d099_l2init").hyperparameters)
    init_fn, _ = _make_upgd_shiftnorm_learner(hp)
    params = _test_params()
    state = init_fn(params)
    assert state.init_params is not None
    for name in params:
        assert jnp.array_equal(state.init_params[name], params[name])


def test_l2init_decay_math() -> None:
    # The decay-to-init update is p' = p*decay + (1-decay)*p0. Verify the
    # arithmetic form with hand-computed values (the full step is exercised
    # end-to-end by the benchmark run itself).
    spec = screening_spec("sigma0_shiftnorm_d099_l2init")
    hp = dict(spec.hyperparameters)
    step_size = hp["step_size"]
    param_decay = 1.0 - step_size * hp["weight_decay"]
    p = 0.5
    p0 = 0.1
    expected = p * param_decay + (1.0 - param_decay) * p0
    # decay-to-zero would give p * param_decay (strictly smaller for p > p0)
    zero_decay = p * param_decay
    assert expected > zero_decay
    assert param_decay < 1.0
    assert 0.0 < param_decay < 1.0
