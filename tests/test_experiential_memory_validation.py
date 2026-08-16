"""Validation hardening for experiential memory (int/float bounds)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.experiential_memory import ExperientialMemoryConfig

_INT32_MAX = 2**31 - 1


class _LyingIntSubclass(int):
    def __int__(self) -> int:  # pragma: no cover
        return 2

    def __index__(self) -> int:  # pragma: no cover
        return 2


class _RaisingIntSubclass(int):
    def __int__(self) -> int:  # pragma: no cover
        raise RuntimeError("conversion hook must not run")

    def __index__(self) -> int:  # pragma: no cover
        raise RuntimeError("conversion hook must not run")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: ExperientialMemoryConfig(
            capacity=v, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=v, key_dim=2, action_dim=1, outcome_dim=1
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=v, action_dim=1, outcome_dim=1
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=v, outcome_dim=1
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=v
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1, top_k=v
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1, min_neighbors=v
        ),
    ],
)
def test_experiential_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1, max_age=v
        ),
    ],
)
def test_experiential_max_age_rejects_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match=r"must be an integer in \[0, 2147483647\]"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"must be an integer in \[0, 2147483647\]"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: ExperientialMemoryConfig(
            capacity=v, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1, max_age=v
        ),
    ],
)
def test_experiential_int_validators_do_not_run_repr_hook(ctor) -> None:
    with pytest.raises(ValueError):
        ctor(_RaisingRepr())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "np_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,  # noqa: E501
        np.ulonglong,
    ],
)
def test_experiential_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = ExperientialMemoryConfig(
        capacity=np_type(8),
        observation_dim=np_type(2),
        key_dim=np_type(2),
        action_dim=np_type(1),
        outcome_dim=np_type(1),
        top_k=np_type(2),
        min_neighbors=np_type(1),
        max_age=np_type(5),
    )
    assert cfg.capacity == 8
    assert type(cfg.capacity) is int
    assert type(cfg.observation_dim) is int
    assert type(cfg.max_age) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: ExperientialMemoryConfig(
            capacity=v, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1
        ),
        lambda v: ExperientialMemoryConfig(
            capacity=4, observation_dim=v, key_dim=2, action_dim=1, outcome_dim=1
        ),
    ],
)
@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1, 10**100]
)
def test_experiential_int_validators_reject_non_integer_and_out_of_range(
    ctor, value: object
) -> None:
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
        ctor(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 4.0, "4", None, -1, _INT32_MAX + 1]
)
def test_experiential_max_age_rejects_non_integer_and_out_of_range(value: object) -> None:
    with pytest.raises(ValueError, match=r"must be an integer in \[0, 2147483647\]"):
        ExperientialMemoryConfig(
            capacity=4, observation_dim=2, key_dim=2, action_dim=1, outcome_dim=1, max_age=value  # type: ignore[arg-type]
        )


def test_experiential_float_validators_reject_nonfinite_and_hostile() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("untrusted ratio hook executed")

    class ClassSpoof:
        @property
        def __class__(self):  # type: ignore[no-untyped-def]
            return float

        def __float__(self) -> float:  # pragma: no cover
            return 0.1

    for field, bad in [
        ("distance_scale", float("nan")),
        ("distance_scale", float("inf")),
        ("distance_scale", 0.0),
        ("distance_scale", -1.0),
        ("distance_scale", HostileFloat(0.5)),
        ("min_similarity", -0.1),
        ("min_similarity", 1.5),
        ("min_similarity", ClassSpoof()),  # type: ignore[arg-type]
        ("min_effective_reliability", 0.0),
        ("min_effective_reliability", 1.5),
        ("max_uncertainty", -0.1),
        ("max_uncertainty", float("nan")),
        ("max_safety_cost", -1.0),
        ("staleness_scale", 0.0),
        ("utility_decay", -0.1),
        ("utility_decay", 1.5),
        ("eviction_utility_weight", -0.1),
        ("recency_scale", 0.0),
        ("recency_scale", float("inf")),
    ]:
        with pytest.raises(ValueError, match=field):
            ExperientialMemoryConfig(
                capacity=4,
                observation_dim=2,
                key_dim=2,
                action_dim=1,
                outcome_dim=1,
                **{field: bad},  # type: ignore[arg-type]
            )


def test_experiential_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="distance_scale"):
        ExperientialMemoryConfig(
            capacity=4,
            observation_dim=2,
            key_dim=2,
            action_dim=1,
            outcome_dim=1,
            distance_scale=HostileFloat(1.0),  # type: ignore[arg-type]
        )


def test_experiential_eviction_weights_require_positive_sum() -> None:
    with pytest.raises(ValueError, match="retention weight"):
        ExperientialMemoryConfig(
            capacity=4,
            observation_dim=2,
            key_dim=2,
            action_dim=1,
            outcome_dim=1,
            eviction_utility_weight=0.0,
            eviction_recency_weight=0.0,
        )


def test_experiential_float_validators_accept_valid_values() -> None:
    cfg = ExperientialMemoryConfig(
        capacity=4,
        observation_dim=2,
        key_dim=2,
        action_dim=1,
        outcome_dim=1,
        distance_scale=1.0,
        min_similarity=0.5,
        min_effective_reliability=0.5,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        staleness_scale=100.0,
        utility_decay=0.9,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )
    assert cfg.distance_scale == 1.0
    assert cfg.min_similarity == 0.5
def test_experiential_dimensions_preflight_without_allocation() -> None:
    # Derived vector sum would overflow signed int32.
    with pytest.raises(ValueError, match="dimensions must fit signed int32"):
        ExperientialMemoryConfig(
            capacity=4,
            observation_dim=_INT32_MAX,
            key_dim=2,
            action_dim=1,
            outcome_dim=1,
        )
    # Scalar-count preflight via capacity * vector overflow.
    with pytest.raises(ValueError, match="scalar count|byte count"):
        ExperientialMemoryConfig(
            capacity=1,
            observation_dim=_INT32_MAX - 18,
            key_dim=1,
            action_dim=1,
            outcome_dim=1,
            top_k=1,
            min_neighbors=1,
        )


def test_experiential_state_preflight_bytes_without_allocation() -> None:
    # Minimal vector=4 (1,1,1,1) -> slot=64, persistent=capacity*64+32.
    last_legal = (2**31 - 1 - 32) // 64
    ExperientialMemoryConfig(
        capacity=last_legal,
        observation_dim=1,
        key_dim=1,
        action_dim=1,
        outcome_dim=1,
    )
    with pytest.raises(ValueError, match="byte count"):
        ExperientialMemoryConfig(
            capacity=last_legal + 1,
            observation_dim=1,
            key_dim=1,
            action_dim=1,
            outcome_dim=1,
        )
    # Non-minimal vector should also be allocation-free.
    with pytest.raises(ValueError, match="byte count|scalar count"):
        ExperientialMemoryConfig(
            capacity=100_000_000,
            observation_dim=2,
            key_dim=2,
            action_dim=2,
            outcome_dim=1,
        )

