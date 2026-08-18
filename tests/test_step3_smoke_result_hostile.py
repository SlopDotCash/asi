"""Complete Step3SmokeResult identity contract: leftover, types, and shapes."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from alberta_framework.steps.step3 import (
    Step3HandoffArrays,
    Step3HordeConfig,
    Step3SmokeResult,
    make_step3_horde,
    run_step3_smoke,
)


def _handoff() -> Step3HandoffArrays:
    return Step3HandoffArrays(
        jnp.ones((8, 3), dtype=jnp.float32),
        jnp.ones((8, 3), dtype=jnp.float32),
        jnp.ones((8, 3), dtype=jnp.float32),
    )


def _legal(**overrides: object) -> Step3SmokeResult:
    payload: dict[str, object] = {
        "config": Step3HordeConfig(),
        "steps": 8,
        "seed": 0,
        "final_window_mse": 0.1,
        "per_demon_metrics_shape": (8, 3, 3),
        "td_errors_shape": (8, 3),
        "finite": True,
        "handoff": _handoff(),
        "horde_config": make_step3_horde(Step3HordeConfig()).to_config(),
    }
    payload.update(overrides)
    return Step3SmokeResult(**payload)  # type: ignore[arg-type]


def test_step3_smoke_result_accepts_canonical_identity() -> None:
    result = _legal()
    assert result.steps == 8
    assert result.seed == 0
    assert result.finite is True
    assert result.per_demon_metrics_shape == (8, 3, 3)
    dumped = json.dumps(
        {"steps": result.steps, "seed": result.seed, "finite": result.finite},
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"seed": 0' in dumped
    assert '"finite": true' in dumped


def test_step3_smoke_binds_real_shapes_handoff_and_serialized_config() -> None:
    config = Step3HordeConfig(gammas=(0.0, 0.5), lamdas=(0.0, 0.5))
    result = run_step3_smoke(config, steps=8, final_window=4, seed=3)
    payload = result.to_dict()

    assert result.per_demon_metrics_shape == (8, 2, 3)
    assert result.td_errors_shape == (8, 2)
    assert result.handoff.observations.shape[0] == result.steps
    assert result.handoff.n_demons == result.config.n_demons
    assert payload["horde_config"] == make_step3_horde(config).to_config()
    json.dumps(payload, allow_nan=False)


def test_step3_smoke_result_rejects_leftover_integer_and_bool_identities() -> None:
    with pytest.raises(ValueError, match="steps must be an integer"):
        _legal(steps=True)
    with pytest.raises(ValueError, match="seed"):
        _legal(seed=True)
    with pytest.raises(ValueError, match="finite must be a boolean"):
        _legal(finite=1)


def test_step3_smoke_result_rejects_leftover_float_and_host_identities() -> None:
    with pytest.raises(ValueError, match="final_window_mse"):
        _legal(final_window_mse=True)
    with pytest.raises(TypeError, match="config must be an exact Step3HordeConfig"):
        _legal(config=None)
    with pytest.raises(TypeError, match="handoff must be an exact Step3HandoffArrays"):
        _legal(handoff=None)
    with pytest.raises(ValueError, match="horde_config must be an exact dict"):
        _legal(horde_config=True)
    with pytest.raises(ValueError, match="per_demon_metrics_shape"):
        _legal(per_demon_metrics_shape=[8, 3, 1])
    with pytest.raises(ValueError, match="td_errors_shape"):
        _legal(td_errors_shape=(7, 3))


def test_step3_smoke_result_rejects_cross_field_mismatches() -> None:
    with pytest.raises(ValueError, match="per_demon_metrics_shape"):
        _legal(per_demon_metrics_shape=(8, 2, 3))
    with pytest.raises(ValueError, match="td_errors_shape"):
        _legal(td_errors_shape=(8, 2))
    with pytest.raises(ValueError, match="handoff row count"):
        _legal(
            handoff=Step3HandoffArrays(
                jnp.ones((7, 3), dtype=jnp.float32),
                jnp.ones((7, 3), dtype=jnp.float32),
                jnp.ones((7, 3), dtype=jnp.float32),
            )
        )
    with pytest.raises(ValueError, match="handoff demon count"):
        _legal(
            handoff=Step3HandoffArrays(
                jnp.ones((8, 3), dtype=jnp.float32),
                jnp.ones((8, 2), dtype=jnp.float32),
                jnp.ones((8, 3), dtype=jnp.float32),
            )
        )
    forged_config = make_step3_horde(Step3HordeConfig()).to_config()
    forged_config["type"] = "forged"
    with pytest.raises(ValueError, match="horde_config does not match"):
        _legal(horde_config=forged_config)
