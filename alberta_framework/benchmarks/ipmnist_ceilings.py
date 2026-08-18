"""Resource contracts for replay and frozen-feature IPMNIST ceilings."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

_METHODS = ("replay", "in_context", "randumb", "ranpac", "prol")

CEILING_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.ipmnist-ceilings.protocol.v1",
        "paper_revisions": (
            "arXiv:2503.20018v1",
            "arXiv:2402.08823v2",
            "arXiv:2307.02251v4",
            "arXiv:2507.12305v1",
        ),
        "methods": _METHODS,
        "pretraining_allowed_but_charged": True,
        "extractor_queries_charged": True,
        "replay_bytes_charged": True,
        "matched_axes": ("seed", "updates", "observations"),
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def _nonnegative(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CeilingResourceLedger:
    """Complete resource charges, including usually hidden ceiling costs."""

    persistent_bytes: int
    replay_bytes: int
    environment_steps: int
    pretraining_steps: int
    model_queries: int
    extractor_queries: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))

    @property
    def total_persistent_bytes(self) -> int:
        return self.persistent_bytes + self.replay_bytes

    @property
    def total_steps(self) -> int:
        return self.environment_steps + self.pretraining_steps

    @property
    def total_model_queries(self) -> int:
        return self.model_queries + self.extractor_queries


@dataclass(frozen=True)
class FrozenFeatureCeiling:
    """Prospective frozen-feature/replay arm declaration."""

    method: Literal["replay", "in_context", "randumb", "ranpac", "prol"]
    feature_dim: int
    replay_capacity: int

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError("method is not a registered ceiling")
        if type(self.feature_dim) is not int or self.feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        _nonnegative("replay_capacity", self.replay_capacity)

    @property
    def mechanism_off(self) -> bool:
        return self.method == "randumb" and self.replay_capacity == 0

    def persistent_replay_bytes(self, *, example_bytes: int) -> int:
        if type(example_bytes) is not int or example_bytes < 0:
            raise ValueError("example_bytes must be a non-negative integer")
        return self.replay_capacity * example_bytes
