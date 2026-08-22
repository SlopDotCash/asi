"""Regression coverage for #2268: integer gates must accept every numpy
integer family via dtype-code derivation, not a hand-enumerated fixed-width
name list.

On LLP64 (64-bit Windows) `np.intc`/`np.uintc` are distinct type objects from
`np.int32`/`np.uint32`, so hand-enumerated sets silently reject valid integers
that neighbour gates accept. The portable fix derives the set from dtype
codes, which names every C-alias spelling on every platform.

These tests pin the derived set's contract without depending on the host
platform's aliasing: every dtype code in the canonical set must be accepted
by the gates, and the set must be exactly the dtype-derived family.
"""

import numpy as np

import alberta_framework.core.stacked_horde as stacked_horde
from alberta_framework.core.stacked_horde import StackedHordeConfig
from alberta_framework.core.types import _require_tracking_interval


def _all_numpy_integer_types() -> set[type]:
    return {np.dtype(code).type for code in "bBhHiIlLqQpP"}


def test_derived_set_accepts_every_dtype_code() -> None:
    for code in "bBhHiIlLqQpP":
        t = np.dtype(code).type
        assert t in stacked_horde._ACTUAL_INT_TYPES, f"{code} ({t}) missing"


def test_derived_set_matches_dtype_family() -> None:
    assert stacked_horde._ACTUAL_INT_TYPES == frozenset(
        {int, *_all_numpy_integer_types()}
    )


def test_intc_and_uintc_are_accepted_through_config() -> None:
    # np.intc/np.uintc are the C int aliases; on Windows they differ from
    # np.int32/np.uint32. The dtype-derived gate must accept them.
    config = StackedHordeConfig(
        n_demons=np.intc(4),
        feature_dim=np.intc(2),
        gammas=(0.9,) * 4,
        lamdas=(0.1,) * 4,
        cumulant_indices=(0, 1, 2, 3),
    )
    assert config.n_demons == 4

    config_u = StackedHordeConfig(
        n_demons=np.uintc(4),
        feature_dim=np.uintc(2),
        gammas=(0.9,) * 4,
        lamdas=(0.1,) * 4,
        cumulant_indices=(0, 1, 2, 3),
    )
    assert config_u.n_demons == 4


def test_tracking_interval_accepts_intc() -> None:
    # The types.py gate (already dtype-derived) accepts intc; stacked_horde
    # must agree with it (the issue's reproduction).
    assert _require_tracking_interval(np.intc(4)) == 4
    assert stacked_horde._require_int32("n", np.intc(4), minimum=0) == 4


def test_platform_alias_is_subsumed() -> None:
    # Whatever the platform aliasing (np.intc is np.int32 on Linux/macOS, a
    # distinct object on Windows), every spelled name resolves into the set.
    for name in ("intc", "uintc", "intp", "uintp", "longlong", "ulonglong"):
        t = getattr(np, name)
        assert t in stacked_horde._ACTUAL_INT_TYPES, name
