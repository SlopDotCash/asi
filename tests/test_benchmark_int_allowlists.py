"""Regression coverage for #2295: benchmark integer allowlists must admit
every numpy integer family via dtype-code derivation.

Two benchmark constants (`_NUMPY_INTEGER_TYPES` in ipmnist_gradual.py and
`_EXACT_NUMPY_INTEGER_TYPES` in causal_map_forager.py) hand-enumerated
fixed-width names and silently rejected C-alias spellings on platforms where
they differ (LLP64/Windows). The #2268 sweep could not reach them because
they are spelled differently from `_ACTUAL_INT_TYPES`.
"""

import numpy as np

from alberta_framework.benchmarks.causal_map_forager import (
    _EXACT_INTEGER_TYPES,
    _EXACT_NUMPY_INTEGER_TYPES,
)
from alberta_framework.benchmarks.ipmnist_gradual import _NUMPY_INTEGER_TYPES


def _all_numpy_integer_types() -> set[type]:
    return {np.dtype(code).type for code in "bBhHiIlLqQpP"}


def test_ipmnist_gradual_covers_every_family() -> None:
    missing = _all_numpy_integer_types() - set(_NUMPY_INTEGER_TYPES)
    assert missing == set()


def test_causal_map_covers_every_family() -> None:
    missing = _all_numpy_integer_types() - set(_EXACT_NUMPY_INTEGER_TYPES)
    assert missing == set()


def test_causal_map_int_not_duplicated() -> None:
    assert _EXACT_INTEGER_TYPES.count(int) == 1
    # int + all numpy families.
    assert len(set(_EXACT_INTEGER_TYPES)) == 11


def test_platform_aliases_subsumed() -> None:
    # intc/uintc/intp/uintp are the C-alias spellings the fixed-width lists
    # could reject on LLP64; the derived sets must accept them everywhere.
    for name in ("intc", "uintc", "intp", "uintp", "longlong", "ulonglong"):
        t = getattr(np, name)
        assert t in _NUMPY_INTEGER_TYPES, f"ipmnist_gradual missing {name}"
        assert t in _EXACT_NUMPY_INTEGER_TYPES, f"causal_map missing {name}"
