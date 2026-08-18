"""Atomic routing for fixed-width dynamic pair-feature banks.

An integrated continual learner commonly deploys a fixed feature vector

``[stable base prefix | discovered pair-feature tail]``.

When the discovered bank is reordered or refreshed, every downstream linear
consumer must move the same columns by *descriptor identity*.  Updating only
one consumer, carrying by slot number, or accepting duplicate descriptors can
silently corrupt learned predictions.  This module provides one fixed-shape,
pure-JAX routing transaction over an arbitrary PyTree of downstream arrays.

Live descriptors are canonical integer pairs ``0 <= left < right < base_dim``.
Inactive slots are exactly ``(-1, -1)``.  Both the old and proposed banks must
contain unique live descriptors.  Invalid transactions fail closed: router
state, counters, and every consumer leaf are returned unchanged, while
machine-readable masks identify duplicate, noncanonical, and out-of-range
slots.

All consumer leaves must contain the full deployed feature width along their
declared feature axis.  The default is the last axis for every leaf.  For any
other layout, callers must provide a PyTree of integer axes with exactly the
same structure as the consumer PyTree; axes are never guessed from shapes.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int

CONFIG_SCHEMA_VERSION = "alberta.feature_bank_router.config.v1"
INACTIVE_DESCRIPTOR = (-1, -1)
_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        *(
            np.dtype(code).type
            for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "p", "P")
        ),
    }
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_derived_int32(name: str, value: int, *, minimum: int = 0) -> int:
    if not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {_INT32_MAX}]")
    return value


def _require_host_count(name: str, value: object, *, minimum: int = 0) -> int:
    """Canonicalize one exact host-only accounting integer without narrowing it."""
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    canonical = operator.index(cast(SupportsIndex, value))
    if canonical < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return canonical


@dataclasses.dataclass(frozen=True)
class FeatureBankRouterConfig:
    """Static shape and descriptor-domain contract for a router.

    ``base_dim`` is both the stable deployed prefix width and the exclusive
    upper bound for pair endpoints.  ``active_slots`` is the fixed capacity of
    the discovered tail; inactive capacity remains represented by ``(-1,-1)``.
    """

    base_dim: int
    active_slots: int

    def __post_init__(self) -> None:
        base_dim = _require_int32("base_dim", self.base_dim, minimum=2)
        active_slots = _require_int32(
            "active_slots",
            self.active_slots,
            minimum=1,
        )
        object.__setattr__(
            self,
            "base_dim",
            base_dim,
        )
        object.__setattr__(
            self,
            "active_slots",
            active_slots,
        )
        _require_derived_int32("total_feature_dim", self.total_feature_dim, minimum=3)
        descriptor_scalars = _require_derived_int32(
            "descriptor_int32_scalars",
            2 * self.active_slots,
            minimum=2,
        )
        router_state_scalars = _require_derived_int32(
            "router_state_scalars",
            descriptor_scalars + 2,
            minimum=4,
        )
        _require_derived_int32(
            "router_state_nbytes",
            4 * router_state_scalars,
            minimum=16,
        )

    @property
    def total_feature_dim(self) -> int:
        """Fixed deployed width of the stable prefix plus dynamic tail."""

        return self.base_dim + self.active_slots

    def to_config(self) -> dict[str, object]:
        """Return a strict JSON-compatible configuration record."""

        return {
            "type": "FeatureBankRouter",
            "schema_version": CONFIG_SCHEMA_VERSION,
            "base_dim": self.base_dim,
            "active_slots": self.active_slots,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> FeatureBankRouterConfig:
        """Reconstruct only the exact versioned configuration schema."""

        if not issubclass(type(config), Mapping):
            raise ValueError("feature-bank router config must be a mapping")
        try:
            payload = dict(config)
        except Exception:
            raise ValueError("feature-bank router config could not be read") from None
        expected_keys = {
            "type",
            "schema_version",
            "base_dim",
            "active_slots",
        }
        if set(payload) != expected_keys:
            raise ValueError("feature-bank router config keys do not match the v1 schema")
        config_type = payload["type"]
        if type(config_type) is not str or config_type != "FeatureBankRouter":
            raise ValueError("feature-bank router config type is invalid")
        schema_version = payload["schema_version"]
        if type(schema_version) is not str or schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError("feature-bank router config schema version is unsupported")
        return cls(
            base_dim=payload["base_dim"],  # type: ignore[arg-type]
            active_slots=payload["active_slots"],  # type: ignore[arg-type]
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("descriptors", "route_count", "generation_count"),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class FeatureBankRouterState:
    """Persistent descriptor identity and monotonic routing counters."""

    descriptors: Int[Array, "active_slots 2"]
    route_count: Int[Array, ""]
    generation_count: Int[Array, ""]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "valid",
        "live_mask",
        "inactive_mask",
        "duplicate_mask",
        "noncanonical_mask",
        "out_of_range_mask",
    ),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class PairDescriptorValidation:
    """Fixed-shape diagnostics for one proposed or deployed descriptor bank."""

    valid: Bool[Array, ""]
    live_mask: Bool[Array, " active_slots"]
    inactive_mask: Bool[Array, " active_slots"]
    duplicate_mask: Bool[Array, " active_slots"]
    noncanonical_mask: Bool[Array, " active_slots"]
    out_of_range_mask: Bool[Array, " active_slots"]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "valid",
        "route_applied",
        "carry_survivors",
        "counter_invalid",
        "descriptors_changed",
        "old_validation",
        "new_validation",
        "source_slots",
        "survivor_mask",
        "new_mask",
        "evicted_mask",
        "survivor_count",
        "new_count",
        "evicted_count",
        "old_live_count",
        "new_live_count",
        "route_count_before",
        "route_count_after",
        "generation_count_before",
        "generation_count_after",
    ),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class FeatureBankRouteDiagnostics:
    """Machine-readable outcome of one atomic route attempt."""

    valid: Bool[Array, ""]
    route_applied: Bool[Array, ""]
    carry_survivors: Bool[Array, ""]
    counter_invalid: Bool[Array, ""]
    descriptors_changed: Bool[Array, ""]
    old_validation: PairDescriptorValidation
    new_validation: PairDescriptorValidation
    source_slots: Int[Array, " active_slots"]
    survivor_mask: Bool[Array, " active_slots"]
    new_mask: Bool[Array, " active_slots"]
    evicted_mask: Bool[Array, " active_slots"]
    survivor_count: Int[Array, ""]
    new_count: Int[Array, ""]
    evicted_count: Int[Array, ""]
    old_live_count: Int[Array, ""]
    new_live_count: Int[Array, ""]
    route_count_before: Int[Array, ""]
    route_count_after: Int[Array, ""]
    generation_count_before: Int[Array, ""]
    generation_count_after: Int[Array, ""]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("state", "consumers", "diagnostics"),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class FeatureBankRouteResult:
    """New router state, atomically remapped consumer PyTree, and diagnostics."""

    state: FeatureBankRouterState
    consumers: Any
    diagnostics: FeatureBankRouteDiagnostics


@dataclasses.dataclass(frozen=True)
class FeatureBankRouterResourceBudget:
    """Exact persistent logical-scalar and byte accounting.

    Diagnostic and route-result buffers are transient and intentionally
    excluded.  Consumer bytes count the supplied managed PyTree exactly,
    including mixed dtypes and arbitrary leading dimensions.
    """

    base_feature_slots: int
    dynamic_feature_slots: int
    total_feature_slots: int
    descriptor_int32_scalars: int
    counter_int32_scalars: int
    router_state_scalars: int
    router_state_nbytes: int
    consumer_leaf_count: int
    consumer_feature_groups: int
    consumer_stable_prefix_scalars: int
    consumer_dynamic_tail_scalars: int
    consumer_total_scalars: int
    consumer_state_nbytes: int
    total_managed_nbytes: int

    def __post_init__(self) -> None:
        """Validate exact formulas while leaving host-only consumer totals unbounded."""
        int32_fields = {
            "base_feature_slots": 2,
            "dynamic_feature_slots": 1,
            "total_feature_slots": 3,
            "descriptor_int32_scalars": 2,
            "counter_int32_scalars": 2,
            "router_state_scalars": 4,
            "router_state_nbytes": 16,
        }
        for name, minimum in int32_fields.items():
            object.__setattr__(
                self,
                name,
                _require_int32(name, getattr(self, name), minimum=minimum),
            )
        host_fields = {
            "consumer_leaf_count": 1,
            "consumer_feature_groups": 0,
            "consumer_stable_prefix_scalars": 0,
            "consumer_dynamic_tail_scalars": 0,
            "consumer_total_scalars": 0,
            "consumer_state_nbytes": 0,
            "total_managed_nbytes": 0,
        }
        for name, minimum in host_fields.items():
            object.__setattr__(
                self,
                name,
                _require_host_count(name, getattr(self, name), minimum=minimum),
            )

        expected = {
            "total_feature_slots": self.base_feature_slots + self.dynamic_feature_slots,
            "descriptor_int32_scalars": 2 * self.dynamic_feature_slots,
            "counter_int32_scalars": 2,
            "router_state_scalars": self.descriptor_int32_scalars
            + self.counter_int32_scalars,
            "router_state_nbytes": 4 * self.router_state_scalars,
            "consumer_stable_prefix_scalars": self.consumer_feature_groups
            * self.base_feature_slots,
            "consumer_dynamic_tail_scalars": self.consumer_feature_groups
            * self.dynamic_feature_slots,
            "consumer_total_scalars": self.consumer_stable_prefix_scalars
            + self.consumer_dynamic_tail_scalars,
            "total_managed_nbytes": self.router_state_nbytes + self.consumer_state_nbytes,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"{name} does not match the derived resource contract")

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact accounting record."""

        return dataclasses.asdict(self)


def _require_descriptor_contract(
    descriptors: Array,
    *,
    active_slots: int,
    location: str,
) -> Array:
    value = jnp.asarray(descriptors)
    if value.shape != (active_slots, 2):
        raise ValueError(f"{location} must have shape ({active_slots}, 2)")
    if value.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"{location} must have dtype int32")
    return value


def _descriptor_validation(
    descriptors: Array,
    *,
    base_dim: int,
) -> PairDescriptorValidation:
    left = descriptors[:, 0]
    right = descriptors[:, 1]
    inactive = (left == INACTIVE_DESCRIPTOR[0]) & (right == INACTIVE_DESCRIPTOR[1])
    in_range = (left >= 0) & (right >= 0) & (left < base_dim) & (right < base_dim)
    live = in_range & (left < right)
    noncanonical = in_range & ~(left < right)
    out_of_range = ~inactive & ~in_range

    equal_pairs = jnp.all(descriptors[:, None, :] == descriptors[None, :, :], axis=-1)
    distinct_slots = ~jnp.eye(descriptors.shape[0], dtype=jnp.bool_)
    duplicates = jnp.any(
        equal_pairs & distinct_slots & live[:, None] & live[None, :],
        axis=1,
    )
    valid = ~jnp.any(duplicates | noncanonical | out_of_range)
    return PairDescriptorValidation(
        valid=valid,
        live_mask=live,
        inactive_mask=inactive,
        duplicate_mask=duplicates,
        noncanonical_mask=noncanonical,
        out_of_range_mask=out_of_range,
    )


def _saturating_increment(value: Array) -> Array:
    return jnp.where(value < _INT32_MAX, value + jnp.int32(1), value)


def _preflight_route_working_set(active_slots: int) -> None:
    """Reject route tensors the host cannot name. Config persist is unchanged."""
    persist_bytes = 4 * (2 * active_slots + 2)
    # Source and proposed router states plus three (slots, slots) bool
    # compare planes: old-bank duplicates, new-bank duplicates, identity match.
    update_working_set_bytes = 2 * persist_bytes + 3 * active_slots * active_slots
    if update_working_set_bytes > _INT32_MAX:
        raise ValueError(
            "feature-bank router update working set byte count must fit signed int32"
        )


class FeatureBankRouter:
    """Atomic fixed-shape router for every downstream feature consumer."""

    def __init__(self, config: FeatureBankRouterConfig):
        self._config = config

    @property
    def config(self) -> FeatureBankRouterConfig:
        """Static router contract."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the exact router configuration."""

        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> FeatureBankRouter:
        """Construct a router from the strict versioned configuration."""

        return cls(FeatureBankRouterConfig.from_config(config))

    def init(
        self,
        descriptors: Array | None = None,
    ) -> FeatureBankRouterState:
        """Initialize counters and an optional fixed-shape descriptor bank.

        Value-level descriptor validity is deliberately checked by
        :meth:`route`, where it can produce a JIT-compatible diagnostic.
        Shape and dtype violations are rejected immediately.
        """

        _preflight_route_working_set(self._config.active_slots)
        if descriptors is None:
            descriptor_array = jnp.tile(
                jnp.asarray(INACTIVE_DESCRIPTOR, dtype=jnp.int32),
                (self._config.active_slots, 1),
            )
        else:
            descriptor_array = _require_descriptor_contract(
                descriptors,
                active_slots=self._config.active_slots,
                location="descriptors",
            )
        return FeatureBankRouterState(
            descriptors=descriptor_array,
            route_count=jnp.array(0, dtype=jnp.int32),
            generation_count=jnp.array(0, dtype=jnp.int32),
        )

    def validate_descriptors(
        self,
        descriptors: Array,
    ) -> PairDescriptorValidation:
        """Return explicit fixed-shape validity diagnostics for one bank."""

        descriptor_array = _require_descriptor_contract(
            descriptors,
            active_slots=self._config.active_slots,
            location="descriptors",
        )
        _preflight_route_working_set(self._config.active_slots)
        return _descriptor_validation(
            descriptor_array,
            base_dim=self._config.base_dim,
        )

    def _consumer_layout(
        self,
        consumers: Any,
        feature_axes: Any | None,
    ) -> tuple[list[Array], jax.tree_util.PyTreeDef, tuple[int, ...]]:
        leaves, tree_definition = jax.tree_util.tree_flatten(consumers)
        if not leaves:
            raise ValueError("consumers must contain at least one array leaf")
        if feature_axes is None:
            raw_axes = [-1] * len(leaves)
        else:
            raw_axes, axes_definition = jax.tree_util.tree_flatten(feature_axes)
            if axes_definition != tree_definition:  # type: ignore[operator]
                raise ValueError(
                    "feature_axes must have exactly the same PyTree structure as consumers"
                )
        arrays: list[Array] = []
        normalized_axes: list[int] = []
        for index, (leaf, raw_axis) in enumerate(zip(leaves, raw_axes, strict=True)):
            value = jnp.asarray(leaf)
            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                raise TypeError(f"feature axis for consumer leaf {index} must be an integer")
            axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
            if axis < 0 or axis >= value.ndim:
                raise ValueError(f"feature axis for consumer leaf {index} is out of range")
            if value.shape[axis] != self._config.total_feature_dim:
                raise ValueError(
                    f"consumer leaf {index} feature axis must have width "
                    f"{self._config.total_feature_dim}"
                )
            arrays.append(value)
            normalized_axes.append(axis)
        return arrays, tree_definition, tuple(normalized_axes)

    def _require_state_contract(self, state: FeatureBankRouterState) -> None:
        _require_descriptor_contract(
            state.descriptors,
            active_slots=self._config.active_slots,
            location="state.descriptors",
        )
        for name, value in (
            ("state.route_count", state.route_count),
            ("state.generation_count", state.generation_count),
        ):
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"{name} must be scalar")
            if array.dtype != jnp.dtype(jnp.int32):
                raise TypeError(f"{name} must have dtype int32")

    def route(
        self,
        state: FeatureBankRouterState,
        consumers: Any,
        new_descriptors: Array,
        *,
        feature_axes: Any | None = None,
        carry_survivors: bool = True,
    ) -> FeatureBankRouteResult:
        """Atomically route every consumer from the old bank to the new bank.

        Surviving live descriptors copy their old column by exact pair
        identity.  Newly born and inactive tail slots are zero.  Evicted
        columns disappear.  The stable prefix is copied without arithmetic.

        ``feature_axes`` is either ``None`` (last axis for every leaf) or a
        PyTree of Python integer axes matching ``consumers`` exactly.  Capture
        non-default axis trees as static closures when using :func:`jax.jit`.

        Setting ``carry_survivors=False`` is the matched no-carry ablation: the
        same valid descriptor transaction and counters are applied, but the
        entire discovered tail is zeroed for every consumer.
        """

        if not isinstance(carry_survivors, bool):
            raise TypeError("carry_survivors must be a Python boolean")
        _preflight_route_working_set(self._config.active_slots)
        self._require_state_contract(state)
        proposed = _require_descriptor_contract(
            new_descriptors,
            active_slots=self._config.active_slots,
            location="new_descriptors",
        )
        consumer_leaves, consumer_tree, axes = self._consumer_layout(
            consumers,
            feature_axes,
        )

        old_validation = _descriptor_validation(
            state.descriptors,
            base_dim=self._config.base_dim,
        )
        new_validation = _descriptor_validation(
            proposed,
            base_dim=self._config.base_dim,
        )
        counter_invalid = (state.route_count < 0) | (state.generation_count < 0)
        valid = old_validation.valid & new_validation.valid & ~counter_invalid

        identity_match = jnp.all(
            proposed[:, None, :] == state.descriptors[None, :, :],
            axis=-1,
        )
        identity_match &= new_validation.live_mask[:, None] & old_validation.live_mask[None, :]
        raw_survivor_mask = new_validation.live_mask & jnp.any(identity_match, axis=1)
        raw_new_mask = new_validation.live_mask & ~raw_survivor_mask
        raw_evicted_mask = old_validation.live_mask & ~jnp.any(identity_match, axis=0)
        raw_source_slots = jnp.argmax(identity_match, axis=1).astype(jnp.int32)
        safe_source_slots = jnp.where(
            raw_survivor_mask,
            raw_source_slots,
            jnp.int32(0),
        )

        routed_leaves: list[Array] = []
        for leaf, axis in zip(consumer_leaves, axes, strict=True):
            prefix_index = [slice(None)] * leaf.ndim
            prefix_index[axis] = slice(0, self._config.base_dim)
            tail_index = [slice(None)] * leaf.ndim
            tail_index[axis] = slice(
                self._config.base_dim,
                self._config.total_feature_dim,
            )
            stable_prefix = leaf[tuple(prefix_index)]
            old_tail = leaf[tuple(tail_index)]
            gathered = jnp.take(old_tail, safe_source_slots, axis=axis)
            mask_shape = [1] * leaf.ndim
            mask_shape[axis] = self._config.active_slots
            survivor_mask = raw_survivor_mask.reshape(mask_shape)
            carried_tail = jnp.where(
                survivor_mask,
                gathered,
                jnp.zeros_like(gathered),
            )
            if not carry_survivors:
                carried_tail = jnp.zeros_like(carried_tail)
            candidate = jnp.concatenate((stable_prefix, carried_tail), axis=axis)
            routed_leaves.append(jnp.where(valid, candidate, leaf))
        routed_consumers = jax.tree_util.tree_unflatten(consumer_tree, routed_leaves)

        descriptors_changed = valid & jnp.any(proposed != state.descriptors)
        candidate_route_count = _saturating_increment(state.route_count)
        candidate_generation_count = jnp.where(
            descriptors_changed,
            _saturating_increment(state.generation_count),
            state.generation_count,
        )
        next_state = FeatureBankRouterState(
            descriptors=jnp.where(valid, proposed, state.descriptors),
            route_count=jnp.where(valid, candidate_route_count, state.route_count),
            generation_count=jnp.where(
                valid,
                candidate_generation_count,
                state.generation_count,
            ),
        )

        survivor_mask = valid & raw_survivor_mask
        new_mask = valid & raw_new_mask
        evicted_mask = valid & raw_evicted_mask
        source_slots = jnp.where(
            survivor_mask,
            raw_source_slots,
            jnp.int32(-1),
        )
        diagnostics = FeatureBankRouteDiagnostics(
            valid=valid,
            route_applied=valid,
            carry_survivors=jnp.asarray(carry_survivors, dtype=jnp.bool_),
            counter_invalid=counter_invalid,
            descriptors_changed=descriptors_changed,
            old_validation=old_validation,
            new_validation=new_validation,
            source_slots=source_slots,
            survivor_mask=survivor_mask,
            new_mask=new_mask,
            evicted_mask=evicted_mask,
            survivor_count=jnp.sum(survivor_mask, dtype=jnp.int32),
            new_count=jnp.sum(new_mask, dtype=jnp.int32),
            evicted_count=jnp.sum(evicted_mask, dtype=jnp.int32),
            old_live_count=jnp.sum(old_validation.live_mask, dtype=jnp.int32),
            new_live_count=jnp.sum(new_validation.live_mask, dtype=jnp.int32),
            route_count_before=state.route_count,
            route_count_after=next_state.route_count,
            generation_count_before=state.generation_count,
            generation_count_after=next_state.generation_count,
        )
        return FeatureBankRouteResult(
            state=next_state,
            consumers=routed_consumers,
            diagnostics=diagnostics,
        )

    def resource_budget(
        self,
        state: FeatureBankRouterState,
        consumers: Any,
        *,
        feature_axes: Any | None = None,
    ) -> FeatureBankRouterResourceBudget:
        """Return exact persistent logical-scalar and byte accounting."""

        self._require_state_contract(state)
        leaves, _, axes = self._consumer_layout(consumers, feature_axes)
        total_scalars = 0
        total_bytes = 0
        feature_groups = 0
        for leaf, axis in zip(leaves, axes, strict=True):
            scalar_count = int(leaf.size)
            groups = scalar_count // int(leaf.shape[axis])
            total_scalars += scalar_count
            feature_groups += groups
            total_bytes += scalar_count * int(leaf.dtype.itemsize)

        descriptor_scalars = 2 * self._config.active_slots
        counter_scalars = 2
        router_state_scalars = descriptor_scalars + counter_scalars
        router_state_nbytes = 4 * router_state_scalars
        stable_prefix_scalars = feature_groups * self._config.base_dim
        dynamic_tail_scalars = feature_groups * self._config.active_slots
        return FeatureBankRouterResourceBudget(
            base_feature_slots=self._config.base_dim,
            dynamic_feature_slots=self._config.active_slots,
            total_feature_slots=self._config.total_feature_dim,
            descriptor_int32_scalars=descriptor_scalars,
            counter_int32_scalars=counter_scalars,
            router_state_scalars=router_state_scalars,
            router_state_nbytes=router_state_nbytes,
            consumer_leaf_count=len(leaves),
            consumer_feature_groups=feature_groups,
            consumer_stable_prefix_scalars=stable_prefix_scalars,
            consumer_dynamic_tail_scalars=dynamic_tail_scalars,
            consumer_total_scalars=total_scalars,
            consumer_state_nbytes=total_bytes,
            total_managed_nbytes=router_state_nbytes + total_bytes,
        )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "INACTIVE_DESCRIPTOR",
    "FeatureBankRouteDiagnostics",
    "FeatureBankRouteResult",
    "FeatureBankRouter",
    "FeatureBankRouterConfig",
    "FeatureBankRouterResourceBudget",
    "FeatureBankRouterState",
    "PairDescriptorValidation",
]
