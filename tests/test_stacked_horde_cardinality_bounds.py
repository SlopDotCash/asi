"""Cardinality bounds for :mod:`alberta_framework.core.stacked_horde`.

The stacked linear Horde must refuse oversized demon axes *before* it walks
per-demon Python sequences or copies attacker-sized lists, mirroring the
12-bit cardinality ceiling already enforced by the peer Horde/history/
option-search modules. These tests pin that contract: they fail against the
unpatched module (which allowed ``n_demons`` up to signed int32) and pass once
the ceiling and length preflights land.
"""

from typing import Any, cast

import pytest

from alberta_framework.core import stacked_horde
from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    _decode_sequence,
    nexting_spec,
)

pytestmark = pytest.mark.unit

_MAX = stacked_horde._MAX_STACKED_HORDE_DEMONS


def _uniform_config(n_demons: int) -> StackedHordeConfig:
    return StackedHordeConfig(
        n_demons=n_demons,
        feature_dim=4,
        gammas=(0.9,) * n_demons,
        lamdas=(0.7,) * n_demons,
        cumulant_indices=(0,) * n_demons,
        step_size=0.05,
    )


def test_max_stacked_horde_demons_is_4096() -> None:
    assert _MAX == 4096
    assert _MAX == (1 << 12)


def test_boundary_n_demons_accepted() -> None:
    cfg = _uniform_config(_MAX)
    assert cfg.n_demons == _MAX
    assert len(cfg.gammas) == _MAX


def test_n_demons_over_ceiling_rejected_before_element_walk() -> None:
    class _Poison:
        """Sentinel that flags if the per-demon element walk ever touches it."""

        touched = False

        def __float__(self) -> float:  # pragma: no cover - must never run
            type(self).touched = True
            raise AssertionError("element walk ran before the n_demons ceiling")

    poison = _Poison()
    over = _MAX + 1
    poison_seq = cast(tuple[float, ...], (poison,) * over)
    with pytest.raises(ValueError, match="n_demons"):
        StackedHordeConfig(
            n_demons=over,
            feature_dim=4,
            gammas=poison_seq,
            lamdas=poison_seq,
            cumulant_indices=(0,) * over,
            step_size=0.05,
        )
    assert _Poison.touched is False


def test_decode_sequence_rejects_oversized_list_before_copy() -> None:
    oversized = [0.0] * (_MAX + 1)
    with pytest.raises(ValueError):
        _decode_sequence("gammas", oversized)


def test_decode_sequence_accepts_boundary_length() -> None:
    boundary = [0.0] * _MAX
    decoded = _decode_sequence("gammas", boundary)
    assert isinstance(decoded, tuple)
    assert len(decoded) == _MAX


def test_decode_sequence_rejects_hostile_list_subclass_without_hooks() -> None:
    class _HostileList(list):  # type: ignore[type-arg]
        len_calls = 0
        iter_calls = 0

        def __len__(self) -> int:  # pragma: no cover - must never run
            type(self).len_calls += 1
            raise AssertionError("__len__ must not run on a list subclass")

        def __iter__(self) -> Any:  # pragma: no cover - must never run
            type(self).iter_calls += 1
            raise AssertionError("__iter__ must not run on a list subclass")

    hostile = _HostileList([0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="actual list or tuple"):
        _decode_sequence("gammas", hostile)
    assert _HostileList.len_calls == 0
    assert _HostileList.iter_calls == 0


def test_from_config_rejects_oversized_serialized_sequence() -> None:
    cfg = _uniform_config(2)
    payload = cfg.to_config()
    payload["gammas"] = [0.9] * (_MAX + 1)
    with pytest.raises(ValueError):
        StackedHordeConfig.from_config(payload)


def test_nexting_spec_rejects_oversized_product() -> None:
    # 65 * 65 = 4225 > 4096: the derived demon grid must be refused before the
    # cumulant-major expansion allocates it.
    cumulant_indices = tuple(range(65))
    gammas = tuple(0.5 for _ in range(65))
    with pytest.raises(ValueError):
        nexting_spec(feature_dim=4, cumulant_indices=cumulant_indices, gammas=gammas)


def test_nexting_spec_accepts_boundary_product() -> None:
    # 64 * 64 = 4096: exactly at the ceiling.
    cumulant_indices = tuple(range(64))
    gammas = tuple(0.5 for _ in range(64))
    cfg = nexting_spec(feature_dim=4, cumulant_indices=cumulant_indices, gammas=gammas)
    assert cfg.n_demons == _MAX
