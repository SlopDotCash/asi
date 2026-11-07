"""Recompute retained development waves without rerunning historical training."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework._strict_json import load_strict_json_object
from alberta_framework.benchmarks.ipmnist_screening import (
    RLSHeadIdentState,
    RLSHeadL2InitState,
    RLSHeadMomentumState,
    _make_rls_head_learner,
    merge_shards,
    screening_spec,
)

_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "ipmnist_screening"


@pytest.mark.parametrize(
    "wave",
    (
        "wave_identmap50_nogate",
        "wave_identmap200_nogate",
        "wave_identmap50_nogate_idet003",
        "wave_identmap50_nogate_idet003_s0075",
        "wave_identmap50_nogate_idet003_s005625",
        "wave_identmap50_r_mom9",
        "wave_identmap50_r_mom99",
        "wave_identmap50_r_idet002",
        "wave_identmap50_r_idet003_s00421875",
        "wave_identmap50_r_idet004_s00421875",
        "wave_identmap50_r_idet003_s0031640625",
        "wave_identmap50_r_idet003_s002373046875",
        "wave_identmap50_r_idet003_s0031640625_r01",
        "wave_identmap50_r_idet003_s0031640625_r02",
        "wave_identmap50_r_idet003_s0031640625_r02_preset003",
        "wave_identmap50_r_idet003_s0031640625_r02_l2init",
        "wave_identmap50_r_idet003_s0031640625_r02_sgdhead",
        "gradual64_r2",
    ),
)
def test_historical_wave_replays_from_registered_shards(wave: str) -> None:
    directory = _OUTPUTS / wave
    retained = load_strict_json_object(directory / "summary.json")
    paths = [directory / Path(row["path"]).name for row in retained["shard_manifest"]]

    recomputed = merge_shards(
        paths,
        control_name=retained["control_name"],
        slope_window=retained["slope_window"],
    )

    # Historical invocations used different working directories. Compare
    # the bound shard bytes and identity independently of their path spelling.
    for summary in (retained, recomputed):
        del summary["created_unix"]
        for shard in summary["shard_manifest"]:
            shard["path"] = Path(shard["path"]).name
    assert recomputed == retained


def test_historical_identmap_l2init_changes_only_the_body_decay_target() -> None:
    parent = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02")
    historical = screening_spec(parent.name + "_l2init")
    params = {
        "w1": jnp.full((4, 3), 0.25),
        "b1": jnp.zeros(3),
        "w2": jnp.full((3, 3), 0.25),
        "b2": jnp.zeros(3),
        "w3": jnp.zeros((3, 2)),
        "b3": jnp.zeros(2),
    }
    x = jnp.ones(4)
    y = jnp.int32(0)
    key = jr.key(0)
    initialize, step = historical.factory(historical.hyperparameters)
    state = initialize(params)
    assert isinstance(state, RLSHeadIdentState)
    assert isinstance(state.inner, RLSHeadL2InitState)
    assert set(state.inner.init_params) == {"w1", "b1", "w2", "b2"}
    actual_params, actual_state, _ = jax.jit(step)(params, state, x, y, key)

    initialize_parent, parent_step = parent.factory(parent.hyperparameters)
    expected_params, expected_state, _ = jax.jit(parent_step)(
        params, initialize_parent(params), x, y, key,
    )
    np.testing.assert_array_equal(actual_params["w1"], params["w1"])
    assert not np.array_equal(expected_params["w1"], params["w1"])
    for name in ("w3", "b3"):
        np.testing.assert_array_equal(actual_params[name], expected_params[name])
    np.testing.assert_array_equal(actual_state.inner.p, expected_state.inner.p)
    np.testing.assert_array_equal(actual_state.inner.wout, expected_state.inner.wout)


@pytest.mark.parametrize("override", ({"decay_to_init": 0.5}, {"step_size": 0.01}))
def test_historical_identmap_l2init_rejects_configuration_drift(
    override: dict[str, float],
) -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02_l2init")
    with pytest.raises(ValueError, match="frozen L2-Init configuration differs"):
        spec.factory({**spec.hyperparameters, **override})


def test_historical_identmap_sgd_head_keeps_covariance_and_applies_decayed_step() -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02_sgdhead")
    params = {
        "w1": jnp.zeros((4, 3)),
        "b1": jnp.zeros(3),
        "w2": jnp.zeros((3, 3)),
        "b2": jnp.zeros(3),
        "w3": jnp.zeros((3, 2)),
        "b3": jnp.zeros(2),
    }
    initialize, step = spec.factory(spec.hyperparameters)
    state = initialize(params)
    assert isinstance(state, RLSHeadIdentState)
    bias = jnp.array([0.4, -0.2])
    state = replace(state, inner=replace(state.inner, wout=state.inner.wout.at[-1].set(bias)))
    _, updated, _ = jax.jit(step)(params, state, jnp.zeros(4), jnp.int32(0), jr.key(0))

    np.testing.assert_allclose(updated.inner.wout[-1], [0.4596, -0.1798], rtol=1e-6)
    np.testing.assert_array_equal(updated.inner.wout[:-1], state.inner.wout[:-1])
    np.testing.assert_array_equal(updated.inner.p, state.inner.p)


@pytest.mark.parametrize("tag", ("mom9", "mom99"))
def test_historical_momentum_changes_only_the_gated_direction(tag: str) -> None:
    parent = screening_spec("rls_head_resid_identmap50_r")
    candidate = screening_spec(parent.name + "_" + tag)
    hp = candidate.hyperparameters
    mu = hp["body_momentum"]
    params = {
        "w1": jnp.full((4, 3), 0.25),
        "b1": jnp.full(3, 0.25),
        "w2": jnp.full((3, 3), 0.25),
        "b2": jnp.full(3, 0.25),
        "w3": jnp.zeros((3, 2)),
        "b3": jnp.zeros(2),
    }
    initialize, step = candidate.factory(hp)
    _, parent_step = parent.factory(parent.hyperparameters)
    state = initialize(params)
    assert isinstance(state, RLSHeadIdentState)
    assert isinstance(state.inner, RLSHeadMomentumState)
    assert set(state.inner.momentum) == {"w1", "b1", "w2", "b2"}
    # Seed the head and utility clock so the test exercises a nonzero
    # gradient with a gate that can pass a change in direction.
    state = replace(state, inner=replace(
        state.inner,
        step=jnp.int32(1000),
        wout=state.inner.wout.at[:-1, 0].set(0.25),
    ))
    step = jax.jit(step)
    parent_step = jax.jit(parent_step)
    utility_decay = hp["utility_decay"]
    decay = 1.0 - hp["step_size"] * hp["weight_decay"]
    for i in range(3):
        inputs = jnp.full(4, 0.2 * (i + 1)), jnp.int32(i % 2), jr.key(0)
        new_params, updated, _ = step(params, state, *inputs)
        parent_params, parent_updated, _ = parent_step(params, state, *inputs)
        np.testing.assert_array_equal(updated.inner.p, parent_updated.inner.p)
        np.testing.assert_array_equal(updated.inner.wout, parent_updated.inner.wout)
        maximum = max(float(updated.inner.utility[name].max()) for name in state.inner.momentum)
        for name, previous in state.inner.momentum.items():
            utility = updated.inner.utility[name]
            np.testing.assert_array_equal(utility, parent_updated.inner.utility[name])
            raw_grad = (utility_decay * state.inner.utility[name] - utility) / (
                (1.0 - utility_decay) * params[name]
            )
            moment = mu * previous + (1.0 - mu) * raw_grad
            np.testing.assert_allclose(updated.inner.momentum[name], moment, atol=1e-7)
            count = updated.inner.step.astype(jnp.float32)
            correction = 1.0 - jnp.float32(mu) ** count
            bias = 1.0 - jnp.float32(utility_decay) ** count
            gate = jax.nn.sigmoid((utility / bias) / (maximum or 1.0))
            np.testing.assert_allclose(
                new_params[name],
                params[name] * decay - hp["step_size"] * (moment / correction) * (1.0 - gate),
                atol=1e-7,
            )
        if i == 2:
            assert any(
                not np.array_equal(new_params[name], parent_params[name])
                for name in state.inner.momentum
            )
        params, state = new_params, updated


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"body_momentum": 1.0 - 2.0**-30}, "float32 rounding"),
        ({"anneal_floor": 0.001}, "fixed-step gated residual body"),
        ({"gate_scale": 0.0}, "fixed-step gated residual body"),
        ({"body_as_meta": 0.001}, "fixed-step gated residual body"),
        ({"polyak_window": 16.0}, "fixed-step gated residual body"),
        ({"snp_shrink": 0.5}, "fixed-step gated residual body"),
    ),
)
def test_historical_momentum_rejects_unsupported_configurations(
    override: dict[str, float], message: str,
) -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_mom9")
    with pytest.raises(ValueError, match=message):
        spec.factory({**spec.hyperparameters, **override})


@pytest.mark.parametrize("keyword", ("lr", "decay_lr"))
def test_historical_momentum_rejects_runtime_schedule(keyword: str) -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_mom9")
    _, step = _make_rls_head_learner(spec.hyperparameters)
    with pytest.raises(ValueError, match="requires a fixed step and decay"):
        step({}, None, jnp.zeros(1), jnp.int32(0), jr.key(0), **{keyword: jnp.float32(0.01)})
