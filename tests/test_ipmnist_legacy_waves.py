"""Recompute retained development waves without rerunning historical training."""

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
        "wave_identmap50_r_idet002",
        "wave_identmap50_r_idet003_s00421875",
        "wave_identmap50_r_idet004_s00421875",
        "wave_identmap50_r_idet003_s0031640625",
        "wave_identmap50_r_idet003_s002373046875",
        "wave_identmap50_r_idet003_s0031640625_r01",
        "wave_identmap50_r_idet003_s0031640625_r02",
        "wave_identmap50_r_idet003_s0031640625_r02_preset003",
        "wave_identmap50_r_idet003_s0031640625_r02_l2init",
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
