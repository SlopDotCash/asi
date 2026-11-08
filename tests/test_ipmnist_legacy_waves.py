"""Recompute retained development waves without rerunning historical training."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework._strict_json import load_strict_json_object
from alberta_framework.benchmarks.ipmnist_screening import (
    RLSHeadBlockState,
    RLSHeadDiagonalState,
    RLSHeadIdentState,
    RLSHeadL2InitState,
    RLSHeadMomentumState,
    _block_rls_gain,
    _diagonal_rls_gain,
    _make_rls_head_learner,
    load_shard,
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
        "wave_identmap50_r_idet003_s0031640625_r02_diaghead",
        "wave_identmap50_r_idet003_s0031640625_r02_gdiag",
        "wave_identmap50_r_idet003_s0031640625_r02_blockhead",
        "wave_identmap50_r_idet003_s0031640625_r02_lowrank",
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


@pytest.mark.parametrize("pooled", (False, True))
def test_diagonal_rls_retains_positive_covariance_under_dominant_excitation(
    pooled: bool,
) -> None:
    p = np.array([1.0 / 0.03, 3.0, 1.0 / 0.03], dtype=np.float32)
    phi = np.array([10000.0, 0.25, 1.0], dtype=np.float32)
    excitation = p.astype(np.float64) * phi.astype(np.float64) ** 2
    denominator = 1.0 + (excitation.sum() if pooled else excitation)
    remaining = 1.0 + (excitation.sum() - excitation if pooled else 0.0)
    expected_gain = p.astype(np.float64) * phi.astype(np.float64) / denominator
    expected_p = p.astype(np.float64) * remaining / denominator

    gain, covariance = jax.jit(_diagonal_rls_gain, static_argnames="pooled")(
        p, phi, pooled=pooled,
    )

    np.testing.assert_allclose(gain, expected_gain, rtol=2e-6)
    np.testing.assert_allclose(covariance, expected_p, rtol=2e-6)
    assert np.all(covariance > 0)
    assert np.all(covariance <= p)


def test_pooled_diagonal_rls_scales_finite_features_before_quadratic_excitation() -> None:
    p = np.full(3, 1.0 / 0.03, dtype=np.float32)
    phi = np.array([1e20, 1e18, 1.0], dtype=np.float32)
    gain, covariance = jax.jit(_diagonal_rls_gain, static_argnames="pooled")(
        p, phi, pooled=True,
    )

    assert np.all(np.isfinite(gain))
    np.testing.assert_allclose(gain[:2], [9.999e-21, 9.999e-23], rtol=2e-6)
    np.testing.assert_allclose(covariance, p * np.array([0.00009999, 0.99990001, 1.0]), rtol=2e-6)

    # Once a diagonal has underflowed, even a huge feature on that axis
    # must leave the other coordinates' excitation at its ordinary scale.
    gain, covariance = jax.jit(_diagonal_rls_gain, static_argnames="pooled")(
        jnp.array([0.0, 3.0, 4.0]), jnp.array([3e38, 1.0, 1.0]), pooled=True,
    )
    np.testing.assert_allclose(gain, [0.0, 3.0 / 8.0, 4.0 / 8.0], rtol=2e-6)
    np.testing.assert_allclose(covariance, [0.0, 3.0 * 5.0 / 8.0, 2.0], rtol=2e-6)


@pytest.mark.parametrize("suffix", ("diaghead", "gdiag"))
def test_live_diagonal_registry_preserves_vector_carry_and_head_reset(suffix: str) -> None:
    name = "rls_head_resid_identmap50_r_idet003_s0031640625_r02_" + suffix
    spec = screening_spec(name + "_stable")
    params = {
        "w1": jnp.zeros((4, 2)),
        "b1": jnp.zeros(2),
        "w2": jnp.zeros((2, 2)),
        "b2": jnp.array([0.5, 0.0]),
        "w3": jnp.zeros((2, 2)),
        "b3": jnp.zeros(2),
    }
    initialize, step = spec.factory(spec.hyperparameters)
    state = initialize(params)
    assert isinstance(state, RLSHeadIdentState)
    assert isinstance(state.inner, RLSHeadDiagonalState)
    assert state.inner.p.shape == (3,)
    inputs = jnp.zeros(4), jnp.int32(1), jr.key(0)
    _, updated, _ = jax.jit(step)(params, state, *inputs)
    expected_gain, expected_p = _diagonal_rls_gain(
        state.inner.p, jnp.array([0.5 / np.sqrt(3), 0.0, 1.0], dtype=jnp.float32),
        pooled=suffix == "gdiag",
    )
    np.testing.assert_allclose(updated.inner.wout[:, 1], expected_gain, rtol=1e-6)
    np.testing.assert_array_equal(updated.inner.wout[:, 0], 0.0)
    np.testing.assert_allclose(updated.inner.p, expected_p, rtol=1e-6)

    # A zero detector threshold forces a covariance reset on this sample.
    _, reset_step = spec.factory({**spec.hyperparameters, "rls_reset_frac": 0.0})
    _, reset, _ = jax.jit(reset_step)(params, state, *inputs)
    np.testing.assert_array_equal(reset.inner.p, state.inner.p)
    np.testing.assert_array_equal(reset.inner.wout, updated.inner.wout)


@pytest.mark.parametrize("suffix", ("diaghead", "gdiag", "blockhead", "lowrank"))
def test_archived_rls_ids_are_replay_only(suffix: str, tmp_path: Path) -> None:
    name = "rls_head_resid_identmap50_r_idet003_s0031640625_r02_" + suffix
    with pytest.raises(ValueError, match="unknown screening config"):
        screening_spec(name)
    directory = _OUTPUTS / f"wave_identmap50_r_idet003_s0031640625_r02_{suffix}"
    retained = directory / f"{name}_seed0.json"
    payload = load_strict_json_object(retained)
    payload["source_provenance"]["git_commit"] = "0" * 40
    forged = tmp_path / "other-source.json"
    forged.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires its original source commit"):
        load_shard(forged)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"rls_head_sgd": 1.0}, "mutually exclusive"),
        ({"rls_head_gdiag": 1.0}, "mutually exclusive"),
        ({"rls_head_block": 1.0}, "mutually exclusive"),
        ({"rls_head_rank": 1.0}, "mutually exclusive"),
        ({"rls_lambda": 0.99}, "requires lambda 1"),
        ({"rls_p_trace_cap": 200.0}, "no trace cap"),
        ({"gate_scale": 0.0}, "fixed-step gated residual body"),
        ({"polyak_window": 16.0}, "fixed-step gated residual body"),
    ),
)
def test_live_diagonal_head_rejects_overlapping_head_and_body_modes(
    override: dict[str, float], message: str,
) -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02_diaghead_stable")
    with pytest.raises(ValueError, match=message):
        spec.factory({**spec.hyperparameters, **override})


@pytest.mark.parametrize("groups", (1, 2, 4))
def test_global_block_head_matches_dense_projection_over_multiple_steps(groups: int) -> None:
    width = -(-5 // groups)
    roots = jnp.tile(jnp.eye(width)[None] / np.sqrt(0.03), (groups, 1, 1))
    rng = np.random.default_rng(12)
    step = jax.jit(_block_rls_gain)
    for _ in range(10):
        phi = np.append(rng.uniform(0.0, 3.0, 4), 1.0).astype(np.float32)
        covariance = np.asarray(roots, dtype=np.float64)
        covariance = covariance @ covariance.transpose(0, 2, 1)
        padded = np.pad(phi.astype(np.float64), (0, groups * width - phi.size))
        features = padded.reshape(groups, width)
        projected = np.einsum("gij,gj->gi", covariance, features)
        denominator = 1.0 + np.sum(features * projected)
        expected = covariance - (
            projected[:, :, None] * projected[:, None, :] / denominator
        )

        gain, roots = step(roots, phi)
        actual = np.asarray(roots) @ np.asarray(roots).transpose(0, 2, 1)

        np.testing.assert_allclose(gain, projected.flatten()[:5] / denominator, atol=2e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-6)
        assert np.all(np.linalg.eigvalsh(actual) > 0.0)


def test_global_block_head_uses_one_innovation_for_all_blocks() -> None:
    gain, roots = jax.jit(_block_rls_gain)(jnp.ones((2, 1, 1)), jnp.ones(2))
    np.testing.assert_allclose(gain, [1.0 / 3.0, 1.0 / 3.0], rtol=1e-6)
    np.testing.assert_allclose(roots[:, 0, 0] ** 2, [2.0 / 3.0, 2.0 / 3.0], rtol=1e-6)


def test_block_joseph_factor_retains_covariance_after_dominant_excitation() -> None:
    roots = jnp.tile(jnp.eye(2)[None] / np.sqrt(0.03), (2, 1, 1))
    gain, updated = jax.jit(_block_rls_gain)(roots, jnp.array([10000.0, 0.0, 1.0]))
    covariance = np.asarray(updated) @ np.asarray(updated).transpose(0, 2, 1)
    assert covariance[0, 0, 0] > 0.0
    expected = (1.0 / 0.03) * (1.0 + 1.0 / 0.03) / (
        1.0 + (1.0 / 0.03) * (1e8 + 1.0)
    )
    np.testing.assert_allclose(covariance[0, 0, 0], expected, rtol=2e-6)
    assert np.all(np.isfinite(gain))
    # The padded coordinate remains an unobserved prior direction.
    np.testing.assert_allclose(covariance[1, 1, 1], 1.0 / 0.03, rtol=1e-6)

    gain, updated = jax.jit(_block_rls_gain)(roots, jnp.array([1e20, 0.0, 1e18, 1.0]))
    covariance = np.asarray(updated) @ np.asarray(updated).transpose(0, 2, 1)
    assert np.all(np.isfinite(gain))
    assert np.all(np.isfinite(covariance))
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_block_registry_preserves_body_updates_and_covariance_reset() -> None:
    parent = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02")
    stable = screening_spec(parent.name + "_blockhead_stable")
    params = {
        "w1": jnp.zeros((4, 4)),
        "b1": jnp.zeros(4),
        "w2": jnp.zeros((4, 4)),
        "b2": jnp.array([0.5, 0.0, 0.0, 0.0]),
        "w3": jnp.zeros((4, 2)),
        "b3": jnp.zeros(2),
    }
    initialize, step = stable.factory(stable.hyperparameters)
    parent_init, parent_step = parent.factory(parent.hyperparameters)
    state = initialize(params)
    assert isinstance(state.inner, RLSHeadBlockState)
    assert state.inner.p.shape == (4, 2, 2)
    args = jnp.zeros(4), jnp.int32(1), jr.key(0)
    actual_params, actual_state, _ = jax.jit(step)(params, state, *args)
    parent_params, parent_state, _ = jax.jit(parent_step)(params, parent_init(params), *args)
    for name in params:
        np.testing.assert_array_equal(actual_params[name], parent_params[name])
    np.testing.assert_allclose(actual_state.inner.wout, parent_state.inner.wout, atol=1e-7)

    _, reset_step = stable.factory({**stable.hyperparameters, "rls_reset_frac": 0.0})
    _, reset_state, _ = jax.jit(reset_step)(params, state, *args)
    np.testing.assert_array_equal(reset_state.inner.p, state.inner.p)
    np.testing.assert_array_equal(reset_state.inner.wout, actual_state.inner.wout)


@pytest.mark.parametrize(
    "override",
    (
        {"rls_head_sgd": 1.0}, {"rls_head_diag": 1.0}, {"rls_head_rank": 8.0},
        {"rls_head_block": 0.5}, {"rls_lambda": 0.999}, {"anneal_floor": 0.001},
        {"polyak_window": 16.0}, {"body_momentum": 0.9},
    ),
)
def test_block_head_rejects_conflicting_modes(override: dict[str, float]) -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02_blockhead_stable")
    with pytest.raises(ValueError):
        spec.factory({**spec.hyperparameters, **override})
