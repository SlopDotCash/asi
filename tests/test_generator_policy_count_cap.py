"""Reject oversized generator-policy tuples before per-element walks hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.resource_manager import (
    _MAX_GENERATOR_POLICIES,
    GeneratorMetaResourceManager,
)


def _policy_kwargs(n: int) -> dict[str, object]:
    return {
        "policy_names": tuple(f"p{i}" for i in range(n)),
        "op_ids": (0,) * n,
        "parent_modes": (0,) * n,
        "replacement_multipliers": (1.0,) * n,
        "promotion_margin_multipliers": (1.0,) * n,
        "candidate_min_age_multipliers": (1.0,) * n,
        "imprint_scales": (1.0,) * n,
    }


def test_generator_policy_cap_constant() -> None:
    assert _MAX_GENERATOR_POLICIES == 4096


def test_generator_manager_accepts_max_policy_count() -> None:
    GeneratorMetaResourceManager(**_policy_kwargs(_MAX_GENERATOR_POLICIES))  # type: ignore[arg-type]


def test_generator_manager_rejects_oversized_policy_count() -> None:
    with pytest.raises(ValueError, match="policy_names length"):
        GeneratorMetaResourceManager(**_policy_kwargs(_MAX_GENERATOR_POLICIES + 1))  # type: ignore[arg-type]


def test_generator_from_config_rejects_oversized_op_ids_before_walk() -> None:
    manager = GeneratorMetaResourceManager(**_policy_kwargs(2))  # type: ignore[arg-type]
    payload = manager.to_config()
    payload["op_ids"] = [0] * (_MAX_GENERATOR_POLICIES + 1)
    with pytest.raises(ValueError, match="op_ids length"):
        GeneratorMetaResourceManager.from_config(payload)
