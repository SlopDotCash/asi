"""Production-facing Step 3 Horde helper tests.

Covers the given-feature Horde facade on real constructors. Invalid
scientific-scalar cases are written to fail on current main (bool,
non-real, non-finite, and out-of-domain values accepted) and pass after
the facade rejects them. Legal endpoints stay constructible.
"""

import json
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.steps.step3 as step3_module
from alberta_framework.core.horde import run_horde_learning_loop
from alberta_framework.steps import (
    Step3HordeConfig,
    build_step2_to_step3_arrays,
    make_step3_horde,
    run_step3_smoke,
)

_INVALID_HORDE_SCALARS: tuple[tuple[str, Any], ...] = (
    ("gammas", (float("nan"),)),
    ("gammas", (float("inf"),)),
    ("gammas", (float("-inf"),)),
    ("gammas", (True,)),
    ("gammas", (False,)),
    ("gammas", ("0.5",)),
    ("gammas", (-0.1,)),
    ("gammas", (1.1,)),
    ("lamdas", (float("nan"),)),
    ("lamdas", (True,)),
    ("lamdas", (1.1,)),
    ("step_size", float("nan")),
    ("step_size", float("inf")),
    ("step_size", True),
    ("step_size", False),
    ("step_size", -1.0),
    ("sparsity", float("nan")),
    ("sparsity", True),
    ("sparsity", -0.1),
    ("sparsity", 1.1),
    ("obgd_kappa", float("nan")),
    ("obgd_kappa", float("inf")),
    ("obgd_kappa", True),
    ("obgd_kappa", 0.0),
    ("obgd_kappa", -1.0),
    ("hidden_sizes", (True,)),
    ("hidden_sizes", (False,)),
    ("hidden_sizes", (0,)),
    ("hidden_sizes", (-1,)),
    ("hidden_sizes", (1.5,)),
    ("hidden_sizes", ("64",)),
)


def test_step2_to_step3_arrays_shift_augmented_observations() -> None:
    raw = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    constructed = jnp.asarray(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    cumulants = jnp.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.25],
            [0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    arrays = build_step2_to_step3_arrays(raw, constructed, cumulants)

    chex.assert_shape(arrays.observations, (3, 6))
    chex.assert_shape(arrays.cumulants, (3, 2))
    chex.assert_shape(arrays.next_observations, (3, 6))
    chex.assert_trees_all_close(
        arrays.observations[0],
        jnp.concatenate([raw[0], constructed[0]]),
    )
    chex.assert_trees_all_close(arrays.next_observations[0], arrays.observations[1])
    chex.assert_trees_all_close(arrays.next_observations[-1], arrays.observations[-1])
    assert arrays.feature_dim == 6
    assert arrays.n_demons == 2
    assert arrays.to_dict()["observations_shape"] == [3, 6]


def test_step3_horde_runs_on_handoff_arrays() -> None:
    raw = jnp.asarray(
        [
            [0.0, 1.0, 0.5],
            [0.2, 0.9, 0.4],
            [0.4, 0.7, 0.3],
            [0.6, 0.5, 0.2],
        ],
        dtype=jnp.float32,
    )
    constructed = jnp.stack([raw[:, 0] * raw[:, 1], raw[:, 1] * raw[:, 2]], axis=1)
    cumulants = jnp.stack([constructed[:, 0], raw[:, 0] + constructed[:, 1]], axis=1)
    arrays = build_step2_to_step3_arrays(raw, constructed, cumulants)

    config = Step3HordeConfig(
        gammas=(0.0, 0.5),
        lamdas=(0.0, 0.2),
        hidden_sizes=(),
        step_size=0.05,
    )
    horde = make_step3_horde(config)
    state = horde.init(arrays.feature_dim, jr.key(0))
    result = run_horde_learning_loop(
        horde,
        state,
        arrays.observations,
        arrays.cumulants,
        arrays.next_observations,
    )

    chex.assert_shape(result.per_demon_metrics, (4, 2, 3))
    chex.assert_shape(result.td_errors, (4, 2))
    chex.assert_tree_all_finite(result.per_demon_metrics)
    chex.assert_tree_all_finite(result.td_errors)
    assert horde.horde_spec.demons[1].gamma == 0.5
    assert horde.horde_spec.demons[1].lamda == 0.2
    assert horde.to_config()["type"] == "HordeLearner"


def test_step3_smoke_is_finite_and_serializable() -> None:
    config = Step3HordeConfig(
        gammas=(0.0, 0.9),
        lamdas=(0.0, 0.8),
        hidden_sizes=(),
        normalizer="ema",
    )
    result = run_step3_smoke(config, steps=16, final_window=4, seed=1)
    payload = result.to_dict()

    assert result.finite
    assert result.final_window_mse >= 0.0
    assert result.per_demon_metrics_shape == (16, 2, 3)
    assert result.td_errors_shape == (16, 2)
    assert payload["config"] == config.to_dict()
    handoff = payload["handoff"]
    horde_config = payload["horde_config"]
    assert isinstance(handoff, dict)
    assert isinstance(horde_config, dict)
    assert handoff["n_demons"] == 2
    assert horde_config["type"] == "HordeLearner"


def test_step3_config_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        make_step3_horde(Step3HordeConfig(gammas=(0.0, 0.9), lamdas=(0.0,)))

    raw = jnp.zeros((2, 3), dtype=jnp.float32)
    constructed = jnp.zeros((3, 1), dtype=jnp.float32)
    cumulants = jnp.zeros((2, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="same number of rows"):
        build_step2_to_step3_arrays(raw, constructed, cumulants)

    with pytest.raises(ValueError, match="at least one demon"):
        build_step2_to_step3_arrays(
            raw,
            jnp.zeros((2, 0), dtype=jnp.float32),
            jnp.zeros((2, 0), dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match="final_window"):
        run_step3_smoke(steps=4, final_window=8)


@pytest.mark.parametrize("field", ("use_obgd", "use_layer_norm"))
def test_step3_config_rejects_non_boolean_algorithm_flags(field: str) -> None:
    payload = Step3HordeConfig().to_dict()
    payload[field] = "false"

    with pytest.raises(ValueError, match=rf"{field} must be a boolean"):
        Step3HordeConfig.from_dict(payload)


def _config_with(**overrides: Any) -> Step3HordeConfig:
    payload: dict[str, Any] = {
        "gammas": (0.0,),
        "lamdas": (0.0,),
    }
    payload.update(overrides)
    return Step3HordeConfig(**payload)


@pytest.mark.parametrize(("field", "value"), _INVALID_HORDE_SCALARS)
def test_step3_horde_scalars_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_step3_horde(_config_with(**{field: value}))


class _SpoofedInt:
    """Mimics ``int`` via ``__class__`` to defeat ``isinstance`` checks."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return int

    def __int__(self) -> int:
        return 3

    def __index__(self) -> int:
        return 3


def test_step3_horde_hidden_sizes_rejects_class_spoofed_integers() -> None:
    with pytest.raises(ValueError, match="hidden_sizes"):
        make_step3_horde(_config_with(hidden_sizes=(_SpoofedInt(),)))


def test_step3_horde_scalars_preserve_legal_boundaries() -> None:
    config = Step3HordeConfig(
        gammas=(0.0, 1.0),
        lamdas=(0.0, 1.0),
        step_size=0.0,
        sparsity=1.0,
        obgd_kappa=2.0,
        use_obgd=False,
    )
    horde = make_step3_horde(config)
    assert horde.horde_spec.demons[0].gamma == 0.0
    assert horde.horde_spec.demons[1].gamma == 1.0
    assert horde.horde_spec.demons[0].lamda == 0.0
    assert horde.horde_spec.demons[1].lamda == 1.0
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = Step3HordeConfig.from_dict(payload)
    assert restored.gammas == (0.0, 1.0)
    assert restored.lamdas == (0.0, 1.0)
    assert restored.step_size == 0.0
    assert restored.sparsity == 1.0
    assert restored.obgd_kappa == 2.0


def test_step3_horde_scalars_canonicalize_nonbuiltin_reals() -> None:
    value = np.float64(0.5)
    config = Step3HordeConfig(
        gammas=(value,),
        lamdas=(value,),
        hidden_sizes=(np.int64(4),),
        step_size=value,
        sparsity=value,
        obgd_kappa=np.float64(2.0),
    )
    horde = make_step3_horde(config)
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert config.gammas == (0.5,)
    assert config.lamdas == (0.5,)
    assert config.hidden_sizes == (4,)
    assert type(payload["gammas"][0]) is float
    assert type(payload["lamdas"][0]) is float
    assert type(payload["hidden_sizes"][0]) is int
    assert type(payload["step_size"]) is float
    assert type(payload["sparsity"]) is float
    assert type(payload["obgd_kappa"]) is float
    assert horde.horde_spec.demons[0].gamma == 0.5


def test_step3_horde_hidden_sizes_stay_in_int32_domain() -> None:
    config = Step3HordeConfig(hidden_sizes=(np.int64(2**31 - 1),))

    assert config.hidden_sizes == (2**31 - 1,)
    assert type(config.hidden_sizes[0]) is int
    with pytest.raises(ValueError, match="hidden_sizes.*int32 max"):
        Step3HordeConfig(hidden_sizes=(2**31,))


@pytest.mark.parametrize(
    "rejected_field",
    ["updates_applied", "head_updates_applied"],
)
def test_step3_smoke_health_gate_reports_refused_updates(
    monkeypatch: pytest.MonkeyPatch,
    rejected_field: str,
) -> None:
    original_run = step3_module.run_horde_learning_loop

    def _refuse_update(*args: Any, **kwargs: Any) -> Any:
        result = original_run(*args, **kwargs)
        if rejected_field == "updates_applied":
            return result.replace(updates_applied=result.updates_applied.at[0].set(False))
        return result.replace(
            head_updates_applied=result.head_updates_applied.at[0, 0].set(False)
        )

    monkeypatch.setattr(step3_module, "run_horde_learning_loop", _refuse_update)

    result = run_step3_smoke(steps=8, final_window=4)

    assert not result.finite
