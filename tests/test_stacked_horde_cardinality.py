"""Regression coverage for #2225: StackedHordeConfig must bound demon count
and per-demon sequence lengths before per-item validation walks.

Oversized demon counts or sequences previously allowed unbounded Python
iteration in the per-demon float32 validation loops.
"""

import pytest

from alberta_framework.core.stacked_horde import (
    _MAX_STACKED_HORDE_DEMONS,
    StackedHordeConfig,
    _decode_sequence,
    nexting_spec,
)


def _config(n_demons: int = 4):
    return {
        "n_demons": n_demons,
        "feature_dim": 2,
        "gammas": (0.9,) * n_demons,
        "lamdas": (0.1,) * n_demons,
        "cumulant_indices": tuple(range(n_demons)),
    }


def test_oversized_n_demons_rejected() -> None:
    with pytest.raises(ValueError, match="at most"):
        StackedHordeConfig(**_config(_MAX_STACKED_HORDE_DEMONS + 1))


def test_nexting_spec_derived_product_rejected_before_grid_build() -> None:
    # nexting_spec derives n_demons = len(indices) * len(gammas); the product
    # must be rejected before the combinatorial grid is materialized (65x65).
    with pytest.raises(ValueError, match="ceiling"):
        nexting_spec(4, tuple(range(65)), (0.0,) * 65)


def test_nexting_spec_normal_product_allowed() -> None:
    cfg = nexting_spec(4, (0, 1, 2), (0.0, 0.5, 0.9, 0.99))
    assert cfg.n_demons == 12


def test_oversized_sequence_rejected() -> None:
    with pytest.raises(ValueError, match="at most"):
        _decode_sequence("gammas", [0.9] * (_MAX_STACKED_HORDE_DEMONS + 1))


def test_boundary_exact_max_allowed() -> None:
    cfg = StackedHordeConfig(**_config(_MAX_STACKED_HORDE_DEMONS))
    assert cfg.n_demons == _MAX_STACKED_HORDE_DEMONS


def test_normal_config_unchanged() -> None:
    cfg = StackedHordeConfig(**_config(4))
    assert cfg.n_demons == 4
    assert cfg.gammas == (0.9, 0.9, 0.9, 0.9)
