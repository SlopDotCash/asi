"""Exact schema, resource, and runtime tests for the Step 3 facade."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps.step3 import (
    Step3HandoffArrays,
    Step3HordeConfig,
    _require_handoff_resources,
    build_step2_to_step3_arrays,
    init_step3_state,
    make_step3_horde,
    run_step3_scan,
    run_step3_smoke,
    step3_predict,
    step3_update,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gammas", [0.0]),
        ("lamdas", [0.0]),
        ("hidden_sizes", [4]),
        ("normalizer", np.str_("none")),
        ("trace_mode", np.str_("accumulating")),
        ("routing", np.str_("shared")),
    ],
)
def test_direct_config_requires_exact_container_and_string_schema(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        Step3HordeConfig(**{field: value})  # type: ignore[arg-type]


def test_config_string_subclass_does_not_execute_equality_hook() -> None:
    class HostileString(str):
        calls = 0

        def __eq__(self, other: object) -> bool:
            type(self).calls += 1
            raise AssertionError("equality hook executed")

        def __hash__(self) -> int:
            return str.__hash__(self)

    with pytest.raises(ValueError, match="routing"):
        Step3HordeConfig(routing=HostileString("shared"))  # type: ignore[arg-type]
    assert HostileString.calls == 0


def test_from_dict_requires_exact_complete_json_schema() -> None:
    payload = Step3HordeConfig().to_dict()

    class DictSubclass(dict[str, object]):
        pass

    invalid_payloads: tuple[tuple[object, str], ...] = (
        (DictSubclass(payload), "actual dict"),
        ({key: value for key, value in payload.items() if key != "routing"}, "fields"),
        ({**payload, "extra": 1}, "fields"),
        ({**payload, "gammas": (0.0, 0.5, 0.9)}, "gammas"),
        ({**payload, "gammas": [np.float32(0.0), 0.5, 0.9]}, "gammas values"),
        ({**payload, "hidden_sizes": [np.int32(4)]}, "hidden_sizes values"),
        ({**payload, "routing": np.str_("shared")}, "routing"),
    )
    for invalid, match in invalid_payloads:
        with pytest.raises(ValueError, match=match):
            Step3HordeConfig.from_dict(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "replacement",
    [
        jnp.ones((2, 1), dtype=jnp.int32),
        jnp.ones((2,), dtype=jnp.float32),
        np.ones((2, 1), dtype=np.float32),
    ],
)
def test_handoff_builder_rejects_dtype_rank_and_array_laundering(
    replacement: object,
) -> None:
    raw = jnp.ones((2, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="raw_observations"):
        build_step2_to_step3_arrays(
            replacement,  # type: ignore[arg-type]
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
        )
    arrays = build_step2_to_step3_arrays(
        raw,
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((2, 1), dtype=jnp.float32),
    )
    assert arrays.feature_dim == 2


def test_handoff_builder_rejects_hostile_object_without_hooks() -> None:
    class Hostile:
        calls = 0

        def __getattribute__(self, name: str) -> Any:
            if name == "calls":
                return object.__getattribute__(self, name)
            type(self).calls += 1
            raise AssertionError("attribute hook executed")

        def __jax_array__(self) -> jax.Array:
            type(self).calls += 1
            raise AssertionError("JAX coercion hook executed")

    hostile = Hostile()
    with pytest.raises(ValueError, match="raw_observations"):
        build_step2_to_step3_arrays(  # type: ignore[arg-type]
            hostile,
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        )
    assert Hostile.calls == 0


def test_direct_handoff_record_enforces_cross_field_identity() -> None:
    observations = jnp.ones((2, 3), dtype=jnp.float32)
    cumulants = jnp.ones((2, 1), dtype=jnp.float32)
    next_observations = jnp.ones((2, 3), dtype=jnp.float32)
    Step3HandoffArrays(observations, cumulants, next_observations)

    with pytest.raises(ValueError, match="next_observations"):
        Step3HandoffArrays(observations, cumulants, next_observations[:1])
    with pytest.raises(ValueError, match="cumulants"):
        Step3HandoffArrays(observations, cumulants[:1], next_observations)


def test_handoff_resource_formula_checks_adjacent_boundary() -> None:
    last_feature_dim = (2**31 - 1) // 8
    assert (
        _require_handoff_resources(
            steps=1,
            raw_dim=last_feature_dim,
            constructed_dim=0,
            n_demons=1,
        )
        == last_feature_dim
    )
    with pytest.raises(ValueError, match="allocation bytes"):
        _require_handoff_resources(
            steps=1,
            raw_dim=last_feature_dim + 1,
            constructed_dim=0,
            n_demons=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", True),
        ("final_window", 1.5),
        ("raw_feature_dim", "2"),
        ("constructed_feature_dim", False),
        ("seed", np.int32(0)),
    ],
)
def test_smoke_host_parameters_follow_exact_policy(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        run_step3_smoke(**{field: value})  # type: ignore[arg-type]


def test_smoke_preflights_all_live_arrays_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("allocation must not start before resource preflight")

    monkeypatch.setattr(jr, "normal", unexpected_allocation)
    with pytest.raises(ValueError, match="smoke allocation bytes"):
        run_step3_smoke(
            steps=10_000_000,
            raw_feature_dim=20,
            constructed_feature_dim=0,
        )


def test_runtime_facade_rejects_narrowing_and_shape_mismatch() -> None:
    config = Step3HordeConfig(gammas=(0.0,), lamdas=(0.0,))
    horde = make_step3_horde(config)
    state = init_step3_state(horde, feature_dim=2, key=jr.key(0))
    feature = jnp.ones((2,), dtype=jnp.float32)
    cumulant = jnp.ones((1,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="features"):
        step3_predict(horde, state, jnp.ones((2,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="next_features"):
        step3_update(horde, state, feature, cumulant, jnp.ones((3,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="cumulants"):
        run_step3_scan(
            horde,
            state,
            feature[None, :],
            jnp.ones((1, 2), dtype=jnp.float32),
            feature[None, :],
        )


def test_runtime_facade_requires_threefry_key() -> None:
    horde = make_step3_horde(Step3HordeConfig(gammas=(0.0,), lamdas=(0.0,)))
    with pytest.raises(ValueError, match="Threefry2x32"):
        init_step3_state(
            horde,
            feature_dim=2,
            key=jr.key(0, impl="rbg"),
        )


def test_make_step3_horde_rejects_hostile_config_without_bool_hook() -> None:
    class HostileConfig:
        calls = 0

        def __bool__(self) -> bool:
            type(self).calls += 1
            raise AssertionError("bool hook executed")

    hostile = HostileConfig()
    with pytest.raises(ValueError, match="config must be"):
        make_step3_horde(hostile)  # type: ignore[arg-type]
    assert HostileConfig.calls == 0
