"""Focused contracts for atomic dynamic feature-bank routing."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.feature_bank_router import (
    CONFIG_SCHEMA_VERSION,
    FeatureBankRouter,
    FeatureBankRouterConfig,
    FeatureBankRouterResourceBudget,
    FeatureBankRouterState,
)

_OLD = jnp.asarray(
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [-1, -1],
    ],
    dtype=jnp.int32,
)
_NEW = jnp.asarray(
    [
        [1, 3],
        [0, 1],
        [2, 3],
        [-1, -1],
    ],
    dtype=jnp.int32,
)

_NUMPY_INTEGER_TYPES = tuple(
    dict.fromkeys(
        np.dtype(code).type
        for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "p", "P")
    )
)


class _RaisingIntegerSpoof:
    @property
    def __class__(self) -> type[int]:
        return int

    def __index__(self) -> int:
        raise AssertionError("__index__ must not be called")

    def __repr__(self) -> str:
        raise AssertionError("__repr__ must not be called")


class _IntegerSubclass(int):
    def __repr__(self) -> str:
        raise AssertionError("__repr__ must not be called")


class _StringSubclass(str):
    pass


class _RaisingStringSpoof:
    @property
    def __class__(self) -> type[str]:
        return str

    def __eq__(self, other: object) -> bool:
        raise AssertionError("__eq__ must not be called")

    def __repr__(self) -> str:
        raise AssertionError("__repr__ must not be called")


def _router() -> FeatureBankRouter:
    return FeatureBankRouter(
        FeatureBankRouterConfig(
            base_dim=4,
            active_slots=4,
        )
    )


def _consumers() -> dict[str, Any]:
    return {
        "behavior_weights": jnp.asarray(
            [
                [1, 2, 3, 4, 10, 20, 30, 40],
                [5, 6, 7, 8, 11, 21, 31, 41],
                [9, 10, 11, 12, 12, 22, 32, 42],
            ],
            dtype=jnp.float32,
        ),
        "control": {
            "q_traces": jnp.asarray(
                [
                    [101, 102],
                    [103, 104],
                    [105, 106],
                    [107, 108],
                    [110, 111],
                    [120, 121],
                    [130, 131],
                    [140, 141],
                ],
                dtype=jnp.float32,
            ),
            "q_weights": jnp.asarray(
                [
                    [201, 202, 203, 204, 210, 220, 230, 240],
                    [205, 206, 207, 208, 211, 221, 231, 241],
                ],
                dtype=jnp.float32,
            ),
        },
    }


def _feature_axes() -> dict[str, Any]:
    return {
        "behavior_weights": -1,
        "control": {
            "q_traces": 0,
            "q_weights": -1,
        },
    }


def _assert_tree_bit_exact(first: Any, second: Any) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree  # type: ignore[operator]
    assert len(first_leaves) == len(second_leaves)
    for left, right in zip(first_leaves, second_leaves, strict=True):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert left_array.tobytes() == right_array.tobytes()


def _assert_state_bit_exact(
    first: FeatureBankRouterState,
    second: FeatureBankRouterState,
) -> None:
    _assert_tree_bit_exact(first, second)


@pytest.mark.unit
def test_reorder_survival_birth_removal_routes_every_consumer_atomically() -> None:
    router = _router()
    state = router.init(_OLD)
    consumers = _consumers()

    result = router.route(
        state,
        consumers,
        _NEW,
        feature_axes=_feature_axes(),
    )

    diagnostics = result.diagnostics
    assert bool(diagnostics.valid)
    assert bool(diagnostics.route_applied)
    np.testing.assert_array_equal(diagnostics.source_slots, [2, 0, -1, -1])
    np.testing.assert_array_equal(diagnostics.survivor_mask, [True, True, False, False])
    np.testing.assert_array_equal(diagnostics.new_mask, [False, False, True, False])
    np.testing.assert_array_equal(diagnostics.evicted_mask, [False, True, False, False])
    assert int(diagnostics.survivor_count) == 2
    assert int(diagnostics.new_count) == 1
    assert int(diagnostics.evicted_count) == 1
    assert int(result.state.route_count) == 1
    assert int(result.state.generation_count) == 1
    np.testing.assert_array_equal(result.state.descriptors, _NEW)

    routed = cast(dict[str, Any], result.consumers)
    original = consumers
    np.testing.assert_array_equal(
        routed["behavior_weights"][:, :4],
        original["behavior_weights"][:, :4],
    )
    np.testing.assert_array_equal(
        routed["control"]["q_weights"][:, :4],
        original["control"]["q_weights"][:, :4],
    )
    np.testing.assert_array_equal(
        routed["control"]["q_traces"][:4, :],
        original["control"]["q_traces"][:4, :],
    )
    np.testing.assert_array_equal(
        routed["behavior_weights"][:, 4:],
        original["behavior_weights"][:, [6, 4, 4, 4]]
        * np.asarray([[1, 1, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        routed["control"]["q_weights"][:, 4:],
        original["control"]["q_weights"][:, [6, 4, 4, 4]]
        * np.asarray([[1, 1, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        routed["control"]["q_traces"][4:, :],
        original["control"]["q_traces"][[6, 4, 4, 4], :]
        * np.asarray([[1], [1], [0], [0]], dtype=np.float32),
    )

    same_generation = router.route(
        result.state,
        result.consumers,
        _NEW,
        feature_axes=_feature_axes(),
    )
    assert int(same_generation.state.route_count) == 2
    assert int(same_generation.state.generation_count) == 1
    assert not bool(same_generation.diagnostics.descriptors_changed)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("invalid", "mask_name", "invalid_slots"),
    [
        (
            jnp.asarray([[0, 1], [0, 1], [1, 3], [-1, -1]], dtype=jnp.int32),
            "duplicate_mask",
            [True, True, False, False],
        ),
        (
            jnp.asarray([[1, 0], [0, 2], [1, 3], [-1, -1]], dtype=jnp.int32),
            "noncanonical_mask",
            [True, False, False, False],
        ),
        (
            jnp.asarray([[0, 4], [0, 2], [1, 3], [-1, -1]], dtype=jnp.int32),
            "out_of_range_mask",
            [True, False, False, False],
        ),
        (
            jnp.asarray([[-1, 2], [0, 2], [1, 3], [-1, -1]], dtype=jnp.int32),
            "out_of_range_mask",
            [True, False, False, False],
        ),
    ],
)
def test_invalid_new_descriptors_fail_closed_without_partial_mutation(
    invalid: jax.Array,
    mask_name: str,
    invalid_slots: list[bool],
) -> None:
    router = _router()
    state = router.init(_OLD)
    consumers = _consumers()

    result = jax.jit(
        lambda router_state, consumer_tree, descriptors: router.route(
            router_state,
            consumer_tree,
            descriptors,
            feature_axes=_feature_axes(),
        )
    )(state, consumers, invalid)

    assert not bool(result.diagnostics.valid)
    assert not bool(result.diagnostics.route_applied)
    np.testing.assert_array_equal(
        getattr(result.diagnostics.new_validation, mask_name),
        invalid_slots,
    )
    np.testing.assert_array_equal(result.diagnostics.source_slots, [-1, -1, -1, -1])
    assert not bool(jnp.any(result.diagnostics.survivor_mask))
    assert not bool(jnp.any(result.diagnostics.new_mask))
    assert not bool(jnp.any(result.diagnostics.evicted_mask))
    _assert_state_bit_exact(result.state, state)
    _assert_tree_bit_exact(result.consumers, consumers)


@pytest.mark.unit
def test_invalid_old_bank_or_counter_also_fails_closed() -> None:
    router = _router()
    duplicate_old = jnp.asarray(
        [[0, 1], [0, 1], [1, 3], [-1, -1]],
        dtype=jnp.int32,
    )
    state = router.init(duplicate_old)
    consumers = _consumers()

    duplicate_result = router.route(
        state,
        consumers,
        _NEW,
        feature_axes=_feature_axes(),
    )

    assert not bool(duplicate_result.diagnostics.valid)
    np.testing.assert_array_equal(
        duplicate_result.diagnostics.old_validation.duplicate_mask,
        [True, True, False, False],
    )
    _assert_state_bit_exact(duplicate_result.state, state)
    _assert_tree_bit_exact(duplicate_result.consumers, consumers)

    negative_counter_state = dataclasses.replace(
        router.init(_OLD),
        route_count=jnp.asarray(-1, dtype=jnp.int32),
    )
    counter_result = router.route(
        negative_counter_state,
        consumers,
        _NEW,
        feature_axes=_feature_axes(),
    )
    assert not bool(counter_result.diagnostics.valid)
    assert bool(counter_result.diagnostics.counter_invalid)
    _assert_state_bit_exact(counter_result.state, negative_counter_state)
    _assert_tree_bit_exact(counter_result.consumers, consumers)


@pytest.mark.unit
def test_no_carry_ablation_keeps_route_identity_and_zeros_only_tail() -> None:
    router = _router()
    state = router.init(_OLD)
    consumers = _consumers()
    carried = router.route(
        state,
        consumers,
        _NEW,
        feature_axes=_feature_axes(),
    )
    no_carry = router.route(
        state,
        consumers,
        _NEW,
        feature_axes=_feature_axes(),
        carry_survivors=False,
    )

    _assert_state_bit_exact(no_carry.state, carried.state)
    np.testing.assert_array_equal(
        no_carry.diagnostics.source_slots,
        carried.diagnostics.source_slots,
    )
    np.testing.assert_array_equal(
        no_carry.diagnostics.survivor_mask,
        carried.diagnostics.survivor_mask,
    )
    assert not bool(no_carry.diagnostics.carry_survivors)
    routed = cast(dict[str, Any], no_carry.consumers)
    np.testing.assert_array_equal(
        routed["behavior_weights"][:, :4],
        consumers["behavior_weights"][:, :4],
    )
    np.testing.assert_array_equal(routed["behavior_weights"][:, 4:], 0.0)
    np.testing.assert_array_equal(
        routed["control"]["q_weights"][:, :4],
        consumers["control"]["q_weights"][:, :4],
    )
    np.testing.assert_array_equal(routed["control"]["q_weights"][:, 4:], 0.0)
    np.testing.assert_array_equal(
        routed["control"]["q_traces"][:4, :],
        consumers["control"]["q_traces"][:4, :],
    )
    np.testing.assert_array_equal(routed["control"]["q_traces"][4:, :], 0.0)


@pytest.mark.unit
def test_route_is_jit_vmap_and_scan_compatible_with_fixed_shapes() -> None:
    router = _router()
    state = router.init(_OLD)
    consumers = _consumers()
    axes = _feature_axes()
    compiled = jax.jit(
        lambda router_state, consumer_tree, descriptors: router.route(
            router_state,
            consumer_tree,
            descriptors,
            feature_axes=axes,
        )
    )

    compiled_result = compiled(state, consumers, _NEW)
    assert bool(compiled_result.diagnostics.valid)
    assert jax.tree_util.tree_structure(compiled_result.consumers) == (  # type: ignore[operator]
        jax.tree_util.tree_structure(consumers)
    )
    for before, after in zip(
        jax.tree_util.tree_leaves(consumers),
        jax.tree_util.tree_leaves(compiled_result.consumers),
        strict=True,
    ):
        assert before.shape == after.shape
        assert before.dtype == after.dtype

    batched_state = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value)),
        state,
    )
    batched_consumers = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value)),
        consumers,
    )
    batched_descriptors = jnp.stack((_NEW, _OLD))
    vmapped = jax.jit(
        jax.vmap(
            lambda router_state, consumer_tree, descriptors: router.route(
                router_state,
                consumer_tree,
                descriptors,
                feature_axes=axes,
            )
        )
    )(batched_state, batched_consumers, batched_descriptors)
    np.testing.assert_array_equal(vmapped.diagnostics.valid, [True, True])
    np.testing.assert_array_equal(vmapped.state.route_count, [1, 1])
    np.testing.assert_array_equal(vmapped.state.generation_count, [1, 0])

    descriptor_sequence = jnp.stack((_NEW, _OLD, _NEW))

    @jax.jit
    def _scan(
        initial_state: FeatureBankRouterState,
        initial_consumers: Mapping[str, Any],
    ) -> tuple[tuple[FeatureBankRouterState, Any], tuple[jax.Array, jax.Array]]:
        def _step(
            carry: tuple[FeatureBankRouterState, Any],
            descriptors: jax.Array,
        ) -> tuple[tuple[FeatureBankRouterState, Any], tuple[jax.Array, jax.Array]]:
            router_state, consumer_tree = carry
            result = router.route(
                router_state,
                consumer_tree,
                descriptors,
                feature_axes=axes,
            )
            return (
                (result.state, result.consumers),
                (
                    result.diagnostics.route_count_after,
                    result.diagnostics.generation_count_after,
                ),
            )

        return jax.lax.scan(
            _step,
            (initial_state, initial_consumers),
            descriptor_sequence,
        )

    (scan_state, scan_consumers), (route_counts, generations) = _scan(
        state,
        consumers,
    )
    np.testing.assert_array_equal(route_counts, [1, 2, 3])
    np.testing.assert_array_equal(generations, [1, 2, 3])
    assert int(scan_state.route_count) == 3
    assert int(scan_state.generation_count) == 3
    assert jax.tree_util.tree_structure(scan_consumers) == (  # type: ignore[operator]
        jax.tree_util.tree_structure(consumers)
    )


@pytest.mark.unit
def test_exact_resource_accounting_and_strict_config_roundtrip() -> None:
    config = FeatureBankRouterConfig(base_dim=4, active_slots=4)
    router = FeatureBankRouter(config)
    state = router.init(_OLD)
    budget = router.resource_budget(
        state,
        _consumers(),
        feature_axes=_feature_axes(),
    )

    assert budget.base_feature_slots == 4
    assert budget.dynamic_feature_slots == 4
    assert budget.total_feature_slots == 8
    assert budget.descriptor_int32_scalars == 8
    assert budget.counter_int32_scalars == 2
    assert budget.router_state_scalars == 10
    assert budget.router_state_nbytes == 40
    assert budget.consumer_leaf_count == 3
    assert budget.consumer_feature_groups == 7
    assert budget.consumer_stable_prefix_scalars == 28
    assert budget.consumer_dynamic_tail_scalars == 28
    assert budget.consumer_total_scalars == 56
    assert budget.consumer_state_nbytes == 224
    assert budget.total_managed_nbytes == 264
    assert budget.to_dict()["total_managed_nbytes"] == 264

    serialized = router.to_config()
    assert serialized == {
        "type": "FeatureBankRouter",
        "schema_version": CONFIG_SCHEMA_VERSION,
        "base_dim": 4,
        "active_slots": 4,
    }
    restored = FeatureBankRouter.from_config(serialized)
    assert restored.config == config
    restored_state = restored.init(_OLD)
    _assert_state_bit_exact(restored_state, state)
    leaves, tree = jax.tree_util.tree_flatten(state)
    checkpoint_roundtrip = jax.tree_util.tree_unflatten(tree, leaves)
    _assert_state_bit_exact(checkpoint_roundtrip, state)

    with pytest.raises(ValueError, match="keys"):
        FeatureBankRouter.from_config({**serialized, "extra": 1})
    with pytest.raises(ValueError, match="schema version"):
        FeatureBankRouter.from_config({**serialized, "schema_version": "unsupported.v2"})
    with pytest.raises(ValueError, match="base_dim"):
        FeatureBankRouterConfig(base_dim=True, active_slots=4)
    with pytest.raises(ValueError, match="active_slots"):
        FeatureBankRouterConfig(base_dim=4, active_slots=0)


@pytest.mark.unit
def test_consumer_layout_and_descriptor_static_contracts_are_strict() -> None:
    router = _router()
    state = router.init(_OLD)
    consumers = _consumers()

    with pytest.raises(ValueError, match="same PyTree"):
        router.route(
            state,
            consumers,
            _NEW,
            feature_axes={"behavior_weights": -1},
        )
    with pytest.raises(ValueError, match="feature axis"):
        router.route(
            state,
            {"bad": jnp.zeros((2, 7), dtype=jnp.float32)},
            _NEW,
        )
    with pytest.raises(TypeError, match="dtype int32"):
        router.route(
            state,
            consumers,
            _NEW.astype(jnp.float32),
            feature_axes=_feature_axes(),
        )


@pytest.mark.unit
def test_router_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="base_dim"):
        FeatureBankRouterConfig(base_dim=True, active_slots=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="active_slots"):
        FeatureBankRouterConfig(base_dim=4, active_slots=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_dim"):
        FeatureBankRouterConfig(base_dim=4.5, active_slots=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="active_slots"):
        FeatureBankRouterConfig(base_dim=4, active_slots=4.5)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("integer_type", _NUMPY_INTEGER_TYPES)
def test_router_config_accepts_and_canonicalizes_all_numpy_integers(
    integer_type: type[np.integer[Any]],
) -> None:
    config = FeatureBankRouterConfig(
        base_dim=integer_type(4),
        active_slots=integer_type(4),
    )
    assert type(config.base_dim) is int
    assert type(config.active_slots) is int
    assert config.base_dim == 4
    assert config.active_slots == 4
    assert json.loads(json.dumps(config.to_config())) == config.to_config()


@pytest.mark.unit
def test_router_config_rejects_derived_feature_width_outside_signed_int32() -> None:
    with pytest.raises(ValueError, match="total_feature_dim"):
        FeatureBankRouterConfig(base_dim=2**31 - 1, active_slots=1)

    boundary = FeatureBankRouterConfig(base_dim=2**31 - 2, active_slots=1)
    assert boundary.total_feature_dim == 2**31 - 1


@pytest.mark.unit
def test_router_config_preflights_derived_state_counts_before_allocation() -> None:
    with pytest.raises(ValueError, match="descriptor_int32_scalars"):
        FeatureBankRouterConfig(base_dim=2, active_slots=1_073_741_824)
    with pytest.raises(ValueError, match="router_state_scalars"):
        FeatureBankRouterConfig(base_dim=2, active_slots=1_073_741_823)
    with pytest.raises(ValueError, match="router_state_nbytes"):
        FeatureBankRouterConfig(base_dim=2, active_slots=268_435_455)

    boundary = FeatureBankRouterConfig(base_dim=2, active_slots=268_435_454)
    assert 4 * (2 * boundary.active_slots + 2) == 2_147_483_640


@pytest.mark.unit
@pytest.mark.parametrize(
    "hostile",
    [_RaisingIntegerSpoof(), _IntegerSubclass(4), np.bool_(True), np.float32(4.0)],
    ids=["class-spoof", "int-subclass", "numpy-bool", "numpy-float"],
)
def test_router_config_rejects_integer_spoofs_without_invoking_hooks(hostile: object) -> None:
    with pytest.raises(ValueError, match="base_dim"):
        FeatureBankRouterConfig(base_dim=hostile, active_slots=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="active_slots"):
        FeatureBankRouterConfig(base_dim=4, active_slots=hostile)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "hostile", "message"),
    [
        ("type", _StringSubclass("FeatureBankRouter"), "type is invalid"),
        ("type", _RaisingStringSpoof(), "type is invalid"),
        ("schema_version", _StringSubclass(CONFIG_SCHEMA_VERSION), "schema version"),
        ("schema_version", _RaisingStringSpoof(), "schema version"),
    ],
    ids=["type-str-subclass", "type-class-spoof", "schema-str-subclass", "schema-class-spoof"],
)
def test_router_config_rejects_string_spoofs_without_invoking_hooks(
    field: str,
    hostile: object,
    message: str,
) -> None:
    serialized = FeatureBankRouterConfig(base_dim=4, active_slots=4).to_config()
    serialized[field] = hostile
    with pytest.raises(ValueError, match=message):
        FeatureBankRouterConfig.from_config(serialized)


@pytest.mark.unit
def test_router_from_config_requires_an_exact_dict_and_exact_schema() -> None:
    payload = FeatureBankRouterConfig(base_dim=4, active_slots=3).to_config()

    class HostileDict(dict[str, object]):
        def __iter__(self) -> Any:
            raise AssertionError("mapping hooks must not run")

        def __repr__(self) -> str:
            raise AssertionError("repr hook must not run")

    for loader in (FeatureBankRouterConfig.from_config, FeatureBankRouter.from_config):
        with pytest.raises(TypeError, match="exact dict"):
            loader(HostileDict(payload))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="keys"):
            loader({key: value for key, value in payload.items() if key != "active_slots"})
        with pytest.raises(ValueError, match="keys"):
            loader({**payload, "unexpected": 1})
        with pytest.raises(ValueError, match="base_dim"):
            loader({**payload, "base_dim": 4.5})
        with pytest.raises(ValueError, match="active_slots"):
            loader({**payload, "active_slots": True})


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "total_feature_slots",
        "descriptor_int32_scalars",
        "counter_int32_scalars",
        "router_state_scalars",
        "router_state_nbytes",
        "consumer_stable_prefix_scalars",
        "consumer_dynamic_tail_scalars",
        "consumer_total_scalars",
        "total_managed_nbytes",
    ],
)
def test_router_resource_budget_rejects_each_inconsistent_derived_count(field: str) -> None:
    router = _router()
    budget = router.resource_budget(
        router.init(_OLD),
        _consumers(),
        feature_axes=_feature_axes(),
    )
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(budget, **{field: getattr(budget, field) + 1})


@pytest.mark.unit
def test_router_resource_budget_keeps_host_only_consumer_counts_unbounded() -> None:
    groups = 10**30
    consumer_nbytes = 10**40
    budget = FeatureBankRouterResourceBudget(
        base_feature_slots=np.int32(2),
        dynamic_feature_slots=np.uint8(1),
        total_feature_slots=np.int64(3),
        descriptor_int32_scalars=np.int16(2),
        counter_int32_scalars=np.uint16(2),
        router_state_scalars=np.int32(4),
        router_state_nbytes=np.int64(16),
        consumer_leaf_count=np.uint8(1),
        consumer_feature_groups=groups,
        consumer_stable_prefix_scalars=2 * groups,
        consumer_dynamic_tail_scalars=groups,
        consumer_total_scalars=3 * groups,
        consumer_state_nbytes=consumer_nbytes,
        total_managed_nbytes=consumer_nbytes + 16,
    )
    assert budget.consumer_total_scalars > 2**31
    assert budget.consumer_state_nbytes > 2**31
    assert all(type(value) is int for value in budget.to_dict().values())
